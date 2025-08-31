#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD EVENTS (in-memory, robust to DSS/Pandas parser quirks)
- Lit le dataset principal en chunks (sans 'columns=' pour éviter les bugs parse_dates/index)
- Garde seulement les colonnes nécessaires en mémoire
- Cast DATMAJ en tz-naive, filtre sur les N derniers mois, déduplique (client, date)
- S'arrête exactement à LIMIT_ROWS (hard stop)
- Donne df_events utilisable directement pour la suite du pipeline
"""

import dataiku
import pandas as pd
from datetime import datetime

# ---------------- CONFIG ----------------
DATASET_MAIN     = "BASE_SCORE_COMPLETE_prepared"  # dataset source
CLIENT_ID_COL    = "NUMTECPRS"                     # id client
TIME_COL         = "DATMAJ"                        # colonne datetime
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"       # produit / item
EXTRA_EVENT_COLS = []                               # ex: ["CANAL", "FAMILLE"]
KEEP_MONTHS      = 24                               # ne garder que les N derniers mois (None pour désactiver)
CHUNKSIZE        = 200_000                          # taille de chunk (ajuste si RAM serrée)
LIMIT_ROWS       = 1_000_000                        # 🔥 limite stricte sur le nombre de lignes en sortie
# -----------------------------------------

print("== BUILD EVENTS (memory) ==", datetime.now())

# 0) Dataset & découverte colonnes sans API non-portable
src = dataiku.Dataset(DATASET_MAIN)
try:
    schema_cols = list(src.get_dataframe(limit=0).columns)  # vide mais avec colonnes
except Exception:
    # fallback vieux DSS
    cfg = src.get_config()
    schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_EVENT_COLS
missing = [c for c in cols_needed if c not in schema_cols]
if missing:
    raise ValueError(f"Colonnes manquantes dans {DATASET_MAIN}: {missing}")

keep_set = set(cols_needed)

# 1) Lecture chunkée SANS 'columns=' ni parse_dates auto
chunks = []
rows_total = 0

for chunk in src.iter_dataframes(
    chunksize=CHUNKSIZE,
    parse_dates=False,          # on parse nous-mêmes
    infer_with_pandas=True
):
    # 1.a) Réduction aux colonnes utiles (ignore extras imprévues)
    inter = [c for c in chunk.columns if c in keep_set]
    if not inter:  # chunk ne contient aucune de nos colonnes (rare)
        continue
    chunk = chunk[inter]

    # 1.b) Nettoyage minimal
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])

    # 1.c) Datetime robuste → tz-aware → tz-naive
    dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
    # drop tz (naive) pour éviter toute comparaison tz-aware vs tz-naive
    dt = dt.dt.tz_convert(None)
    chunk[TIME_COL] = dt
    chunk = chunk.dropna(subset=[TIME_COL])

    # 1.d) Filtre temporel (derniers N mois)
    if KEEP_MONTHS is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS)
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    if chunk.empty:
        continue

    # 1.e) Dédup (client, date) → garder la + récente
    chunk = (
        chunk.sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
    )

    if chunk.empty:
        continue

    # 1.f) Application stricte de la limite (exacte)
    remaining = LIMIT_ROWS - rows_total
    if remaining <= 0:
        print(f"STOP: limite {LIMIT_ROWS:,} atteinte.")
        break

    if len(chunk) > remaining:
        # on tronque le chunk pour respecter exactement LIMIT_ROWS
        chunk = chunk.iloc[:remaining]

    chunks.append(chunk)
    rows_total += len(chunk)

    # Log progression
    print(f"   + {len(chunk):,} rows (cumulative={rows_total:,})")

    if rows_total >= LIMIT_ROWS:
        print(f"STOP: limite {LIMIT_ROWS:,} atteinte.")
        break

# 2) Concat final
if chunks:
    df_events = pd.concat(chunks, ignore_index=True)
else:
    df_events = pd.DataFrame(columns=cols_needed)

# 3) Résumé
print("== DONE ==")
print(f"Final events shape : {df_events.shape}")
print(df_events.head(10))

# À ce stade :
# - df_events contient uniquement NUMTECPRS, DATMAJ, SOUSCRIPTION_PRODUIT_1M (+ extras éventuels)
# - Chaque (client, date) est unique, sur les {KEEP_MONTHS} derniers mois
# - Taille <= LIMIT_ROWS
# - Prêt à être ingéré par votre SequenceBuilder / pipeline_core.run_training


# ===============================================================
# 2) FS (profil) => SAVE EVENTS => CONFIG => TRAIN => METRICS
# ===============================================================

import dataiku
from datetime import datetime
import pandas as pd

# --- Toolkit (ta lib) ---
from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig
from t4rec_toolkit.pipeline_core import (
    blank_config,
    run_training,
    evaluate_topk_metrics_nbo,   # déjà importé dans pipeline_core, ici pour affichage si besoin
)

# ------------ PARAMS ------------
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"   # table profil (la même que tout à l’heure)
EVENTS_TMP   = "TMP_T4REC_EVENTS_RAM"          # dataset temporaire pour les événements
CLIENT_ID_COL = "NUMTECPRS"
TIME_COL      = "DATMAJ"
PRODUCT_COL   = "SOUSCRIPTION_PRODUIT_1M"

# Candidats profil (fallback si FS indispo / vide)
PROFILE_CATEGORICAL_CANDIDATES = ["MEILLEURE_CARTE_DETENUE_M", "LIBFAMCSP", "CONNAISSANCE_MIF"]
PROFILE_SEQUENCE_CANDIDATES    = ["AGE"]

# Cible: exclure la “non-souscription”
TARGET_EXCLUDE = ["aucune_souscription"]

# Fenêtre temporelle (longueur de séquence)
MONTHS_LOOKBACK = 24

# Taille modèle & entraînement (anti-OOM friendly)
D_MODEL  = 512      # ↓ si OOM : 384 ou 256
N_HEADS  = 8        # ↓ si OOM : 4
N_LAYERS = 4        # ↓ si OOM : 2–3
BATCH_SIZE = 64     # ↓ si OOM : 32 ou 16
EPOCHS     = 10     # ↑ plus tard si OK
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.2

print("\n== STEP A: FS (profil) ==", datetime.now())
seq_cols_fs, cat_cols_fs = [], []
try:
    fs_cfg = FeatureSelectorConfig(
        sample_size=50_000,            # rapide
        total_feature_cap=20,
        top_k_sequence=10,             # numériques profil
        top_k_categorical=8,           # catégorielles profil
        compute_model_importance=True,
        rf_n_estimators=100,
        corr_threshold=0.85,
        chunk_size=20_000,
        downcast_dtypes=True,
        correlation_batch_size=100,
        gc_frequency=50,
        report_dataset=None,
        verbose=True, progress=True,
    )
    fs = FeatureSelector(fs_cfg)
    fs.fit(DATASET_MAIN, None, PRODUCT_COL)         # table + target
    selected = fs.get_selected_features()
    seq_cols_fs = selected.get("sequence_cols", [])
    cat_cols_fs = selected.get("categorical_cols", [])
    print("FS OK.")
except Exception as e:
    print(f"FS facultative – on continue sans: {e}")

# Si la FS n’a rien rendu, on garde les colonnes “sûres”
profile_categorical_cols = [c for c in PROFILE_CATEGORICAL_CANDIDATES if (not cat_cols_fs) or (c in cat_cols_fs)]
profile_sequence_cols    = [c for c in PROFILE_SEQUENCE_CANDIDATES    if (not seq_cols_fs) or (c in seq_cols_fs)]

print("Profil retenu — cat:", profile_categorical_cols)
print("Profil retenu — num:", profile_sequence_cols)

# --------------------------------------------------------------
print("\n== STEP B: SAVE df_events -> Dataiku dataset ==", datetime.now())
# df_events vient du bloc précédent
assert isinstance(df_events, pd.DataFrame) and not df_events.empty, "df_events est vide ou absent"
# On force l’ordre des colonnes utiles (si extras présents, on les garde dans la fin)
base_cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
extras = [c for c in df_events.columns if c not in base_cols]
df_events = df_events[base_cols + extras]

# Ecriture/écrasement avec schéma
dataiku.Dataset(EVENTS_TMP).write_with_schema(df_events)
print(f"Saved events -> {EVENTS_TMP} : {df_events.shape}")

# --------------------------------------------------------------
print("\n== STEP C: CONFIG ==", datetime.now())

config = blank_config()

# Données (événements obligatoires)
config["data"]["events_dataset"]   = EVENTS_TMP
config["data"]["client_id_col"]    = CLIENT_ID_COL
config["data"]["event_time_col"]   = TIME_COL
config["data"]["product_col"]      = PRODUCT_COL
config["data"]["event_extra_cols"] = []  # si tu as des extras dans df_events (ex: canal), liste-les ici

# Données (profil facultatif)
config["data"]["dataset_name"]                 = DATASET_MAIN
config["data"]["profile_join_key"]             = CLIENT_ID_COL
config["data"]["profile_categorical_cols"]     = profile_categorical_cols
config["data"]["profile_sequence_cols"]        = profile_sequence_cols

# Fenêtre temporelle
config["sequence"]["months_lookback"]          = MONTHS_LOOKBACK
config["sequence"]["time_granularity"]         = "M"
config["sequence"]["min_events_per_client"]    = 1
config["sequence"]["target_horizon"]           = 1
config["sequence"]["pad_value"]                = 0
config["sequence"]["build_target_from_events"] = True

# Features / cible
config["features"]["exclude_target_values"] = TARGET_EXCLUDE
config["features"]["merge_rare_threshold"]  = 200
config["features"]["other_class_name"]      = "AUTRES_PRODUITS"

# Modèle
config["model"]["d_model"]             = D_MODEL
config["model"]["n_heads"]             = N_HEADS
config["model"]["n_layers"]            = N_LAYERS
config["model"]["dropout"]             = 0.10
config["model"]["max_sequence_length"] = MONTHS_LOOKBACK
config["model"]["vocab_size"]          = 2000

# Entraînement
config["training"]["batch_size"]      = BATCH_SIZE
config["training"]["num_epochs"]      = EPOCHS
config["training"]["learning_rate"]   = LEARNING_RATE
config["training"]["weight_decay"]    = 1e-4
config["training"]["val_split"]       = VAL_SPLIT
config["training"]["class_weighting"] = True
config["training"]["gradient_clip"]   = 1.0
config["training"]["optimizer"]       = "adamw"

# Sorties (mets None si tu ne veux rien écrire)
config["outputs"]["features_dataset"]        = "T4REC_FEATURES"
config["outputs"]["predictions_dataset"]     = "T4REC_PREDICTIONS"
config["outputs"]["metrics_dataset"]         = "T4REC_METRICS"
config["outputs"]["model_artifacts_dataset"] = "T4REC_MODEL"
config["outputs"]["local_dir"]               = "output"

# Runtime
config["runtime"]["verbose"]  = True
config["runtime"]["progress"] = True
config["runtime"]["seed"]     = 42

print(f"Archi: {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D")
print("OK.")

# --------------------------------------------------------------
print("\n== STEP D: TRAIN ==", datetime.now())
results = run_training(config)
print("Train OK.")

# --------------------------------------------------------------
print("\n== STEP E: METRICS ==", datetime.now())

# Métriques standard
m = results.get("metrics", {})
for k in ["accuracy", "precision", "recall", "f1"]:
    if k in m:
        print(f"{k:9s} : {m[k]:.4f}")

# Top-K (si déjà calculées par pipeline_core, elles sont aussi écrites dans T4REC_METRICS)
saved = results.get("saved_datasets", {})
if "metrics" in saved:
    try:
        met_df = dataiku.Dataset(config["outputs"]["metrics_dataset"]).get_dataframe()
        topk_view = met_df[met_df["metric_type"]=="topk_nbo"].sort_values(["k_value","metric_name"])
        print("\nTop-K (aperçu, depuis dataset):")
        print(topk_view.head(20))
    except Exception as e:
        print(f"Lecture métriques dataset impossible (non bloquant): {e}")
else:
    # Affichage léger à partir de l’objet results si besoin
    preds = results.get("predictions", {})
    raw = preds.get("raw_outputs", None)
    y   = preds.get("true_classes", None)
    if raw is not None and y is not None:
        # petite inférence sur le mapping inverse depuis pipeline (si renvoyé)
        print("Top-K rapide (calcul local)…")
        import numpy as np
        # sans mapping explicite des noms produits ici, on reste sur indices
        inv_map = {i: f"class_{i}" for i in range(raw.shape[1])}
        tk, _ = evaluate_topk_metrics_nbo(predictions=raw, targets=y, inverse_target_mapping=inv_map, k_values=[1,3,5])
        for k,v in tk.items():
            print(f"K={k}: P@K={v['Precision@K']:.4f} | R@K={v['Recall@K']:.4f} | F1@K={v['F1@K']:.4f} | NDCG@K={v['NDCG@K']:.4f}")

print("\n== DONE ==")






