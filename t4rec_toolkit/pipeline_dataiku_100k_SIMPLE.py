#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# NOTEBOOK T4REC — PIPELINE TEMPOREL (FS => CONFIG => TRAIN => METRICS)
# Optimisé pour éviter OOM (séquence plus courte, modèle plus léger,
# batch plus petit, filtrage amont des événements).
# ===============================================================

# -------- Imports standards --------
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# -------- Imports Dataiku --------
import dataiku

# -------- Toolkit interne --------
from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig
from t4rec_toolkit.pipeline_core import (
    blank_config,
    run_training,
)

# ============== PARAMÈTRES HAUT NIVEAU ==============
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"     # Table principale (~6M lignes)
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"         # Vue événements (sera créée)
CLIENT_ID_COL = "CLIENT_ID"                       # Identifiant client
TIME_COL = "DATMAJ"                               # Colonne temporelle (mois)
PRODUCT_COL = "SOUSCRIPTION_PRODUIT_1M"           # Produit souscrit
EXTRA_EVENT_COLS = []                             # Éventuels extras (ex: ["CANAL","FAMILLE"])
PROFILE_CATEGORICAL_CANDIDATES = [
    "MEILLEURE_CARTE_DETENUE_M", "LIBFAMCSP", "CONNAISSANCE_MIF"
]
PROFILE_SEQUENCE_CANDIDATES = ["AGE"]
TARGET_EXCLUDE = ["aucune_souscription"]
RARE_MIN_COUNT = 1000   # seuil augmenté (anti-OOM)

# ============== AFFICHE ENTÊTE ==============
print("PIPELINE T4REC — DIMENSION TEMPORELLE OPTIMISÉ")
print("=" * 80)
print(f"Date/Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset principal          : {DATASET_MAIN}")
print(f"Dataset événements (vue)   : {DATASET_EVENTS}")
print("-" * 80)

# ============== 1) FEATURE SELECTION SUR LE PROFIL ==============
print("\n1) FEATURE SELECTION (profil)")
print("-" * 80)
try:
    fs_config = FeatureSelectorConfig(
        sample_size=50_000,
        total_feature_cap=20,
        top_k_sequence=12,
        top_k_categorical=8,
        compute_model_importance=True,
        rf_n_estimators=100,
        corr_threshold=0.85,
        chunk_size=20_000,
        downcast_dtypes=True,
        correlation_batch_size=100,
        gc_frequency=50,
        report_dataset=None,
        verbose=True,
        progress=True,
    )
    print("→ Lancement FS…")
    selector = FeatureSelector(fs_config)
    selector.fit(DATASET_MAIN, None, PRODUCT_COL)
    selected = selector.get_selected_features()
    seq_cols_fs = selected.get("sequence_cols", [])
    cat_cols_fs = selected.get("categorical_cols", [])
    print(f"   Numériques proposées : {seq_cols_fs}")
    print(f"   Catégorielles proposées : {cat_cols_fs}")
except Exception as e:
    print(f"⚠️ FS échouée (non bloquant): {e}")
    seq_cols_fs, cat_cols_fs = [], []

profile_categorical_cols = [c for c in PROFILE_CATEGORICAL_CANDIDATES if c in cat_cols_fs or len(cat_cols_fs) == 0]
profile_sequence_cols    = [c for c in PROFILE_SEQUENCE_CANDIDATES if c in seq_cols_fs or len(seq_cols_fs) == 0]

print("\n   → CATEGOS profil retenues :", profile_categorical_cols)
print("   → NUMÉRIQUES profil retenues :", profile_sequence_cols)

# ============== 2) CONSTRUCTION VUE “ÉVÉNEMENTS” ==============
print("\n2) CONSTRUCTION VUE ÉVÉNEMENTS")
print("-" * 80)
try:
    df_main = dataiku.Dataset(DATASET_MAIN).get_dataframe()
    print(f"   Table principale : {df_main.shape}")

    # Vérifie colonnes
    missing_cols = [c for c in [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] if c not in df_main.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols}")

    events = df_main[[CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + [c for c in EXTRA_EVENT_COLS if c in df_main.columns]].copy()
    events = events.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    events[TIME_COL] = pd.to_datetime(events[TIME_COL], errors="coerce")
    events = events.dropna(subset=[TIME_COL])

    # ⚡ Optimisation mémoire : limiter aux 18 derniers mois
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=18)
    events = events[events[TIME_COL] >= cutoff]

    # ⚡ Option: échantillonner max 300k clients
    MAX_CLIENTS = 300_000
    if events[CLIENT_ID_COL].nunique() > MAX_CLIENTS:
        keep_clients = events[CLIENT_ID_COL].drop_duplicates().sample(MAX_CLIENTS, random_state=42)
        events = events[events[CLIENT_ID_COL].isin(keep_clients)]

    # Nettoyage doublons → garde la dernière souscription par (client, mois)
    events = (events
              .sort_values([CLIENT_ID_COL, TIME_COL])
              .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))

    # Sauvegarde dataset
    dataiku.Dataset(DATASET_EVENTS).write_with_schema(events)
    print(f"   ✅ Vue événements sauvegardée : {events.shape}")
except Exception as e:
    print(f"❌ Erreur construction événements : {e}")
    raise

# ============== 3) CONFIGURATION PIPELINE ==============
print("\n3) CONFIGURATION")
print("-" * 80)
config = blank_config()

# Données
config["data"]["events_dataset"]             = DATASET_EVENTS
config["data"]["client_id_col"]              = CLIENT_ID_COL
config["data"]["event_time_col"]             = TIME_COL
config["data"]["product_col"]                = PRODUCT_COL
config["data"]["event_extra_cols"]           = EXTRA_EVENT_COLS
config["data"]["dataset_name"]               = DATASET_MAIN
config["data"]["profile_join_key"]           = CLIENT_ID_COL
config["data"]["profile_categorical_cols"]   = profile_categorical_cols
config["data"]["profile_sequence_cols"]      = profile_sequence_cols

# Séquences (⚡ raccourcies à 12 mois)
config["sequence"]["months_lookback"]        = 12
config["sequence"]["time_granularity"]       = "M"
config["sequence"]["min_events_per_client"]  = 1
config["sequence"]["target_horizon"]         = 1
config["sequence"]["pad_value"]              = 0
config["sequence"]["build_target_from_events"]= True

# Features
config["features"]["exclude_target_values"]  = TARGET_EXCLUDE
config["features"]["merge_rare_threshold"]   = RARE_MIN_COUNT
config["features"]["other_class_name"]       = "AUTRES_PRODUITS"

# Modèle (⚡ allégé)
config["model"]["d_model"]                   = 256
config["model"]["n_heads"]                   = 8
config["model"]["n_layers"]                  = 4
config["model"]["dropout"]                   = 0.15
config["model"]["max_sequence_length"]       = 12
config["model"]["vocab_size"]                = 2000

# Entraînement
config["training"]["batch_size"]             = 32
config["training"]["num_epochs"]             = 20
config["training"]["learning_rate"]          = 5e-4
config["training"]["weight_decay"]           = 1e-4
config["training"]["val_split"]              = 0.10
config["training"]["class_weighting"]        = True
config["training"]["gradient_clip"]          = 1.0
config["training"]["optimizer"]              = "adamw"

# Sorties
config["outputs"]["features_dataset"]        = "T4REC_FEATURES"
config["outputs"]["predictions_dataset"]     = "T4REC_PREDICTIONS"
config["outputs"]["metrics_dataset"]         = "T4REC_METRICS"
config["outputs"]["model_artifacts_dataset"] = "T4REC_MODEL"
config["outputs"]["local_dir"]               = "output"

# Runtime
config["runtime"]["verbose"]                 = True
config["runtime"]["progress"]                = True
config["runtime"]["seed"]                    = 42

print(f"   Archi : {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D")
print(f"   Séquence : {config['sequence']['months_lookback']} mois")
print(f"   Profil : cat={config['data']['profile_categorical_cols']} | num={config['data']['profile_sequence_cols']}")
print(f"   Classes rares < {config['features']['merge_rare_threshold']} regroupées en {config['features']['other_class_name']}")

# ============== 4) ENTRAÎNEMENT ==============
print("\n4) ENTRAÎNEMENT")
print("-" * 80)
try:
    results = run_training(config)
    print("   ✅ Entraînement terminé.")
except Exception as e:
    print(f"❌ ERREUR entraînement: {e}")
    raise

# ============== 5) RÉSULTATS ==============
print("\n5) RÉSULTATS & MÉTRIQUES")
print("-" * 80)
metrics = results.get("metrics", {})
print("   → Métriques validation :")
for k in ["accuracy", "precision", "recall", "f1"]:
    if k in metrics:
        print(f"     - {k:9s} : {metrics[k]:.4f}")

mi = results.get("model_info", {})
di = results.get("data_info", {})
print("\n   → Modèle :", mi.get("architecture", "N/A"))
print("   → Paramètres :", f"{mi.get('total_params', 0):,}")
print("   → Données :")
print(f"     - Clients : {di.get('n_clients', 'N/A')}")
print(f"     - Longueur séq. : {di.get('seq_len', 'N/A')}")
print(f"     - Classes cible : {di.get('n_classes', 'N/A')}")
print(f"     - Features emb. : {di.get('features', [])}")

print("\nPIPELINE TERMINÉ ✅")
