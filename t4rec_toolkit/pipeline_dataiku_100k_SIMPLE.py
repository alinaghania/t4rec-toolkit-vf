#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# BUILD EVENTS IN-MEMORY ➜ TRAIN (no output dataset)
# - Source unique : BASE_SCORE_COMPLETE_prepared (multi-lignes/client)
# - Vue événements en RAM : [NUMTECPRS, DATMAJ, SOUSCRIPTION_PRODUIT_1M, ...]
# - Patch du loader pour injecter le DataFrame au pipeline
# ===============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import dataiku

# --- toolkit ---
from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig
import t4rec_toolkit.pipeline_core as pc  # on va patcher _load_events_df
from t4rec_toolkit.pipeline_core import blank_config, run_training

print("== BUILD EVENTS (memory) ==", datetime.now())

# ------------------------
# 1) Paramètres principaux
# ------------------------
DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"   # table source
CLIENT_ID_COL  = "NUMTECPRS"                      # ✅ remplacé
TIME_COL       = "DATMAJ"
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []                             # ex: ["CANAL", "FAMILLE"]

# Limites mémoire (adapter si besoin)
CHUNKSIZE      = 250_000        # lecture par morceaux
KEEP_MONTHS    = 36             # ne garder que les 36 derniers mois (réduit la RAM)
MAX_ROWS_HINT  = None           # par ex 3_000_000 pour couper si OOM

# -----------------------------
# 2) Construire events_df (RAM)
# -----------------------------
src = dataiku.Dataset(DATASET_MAIN)

# On ne lit pas tout: on streame uniquement les colonnes utiles
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_EVENT_COLS

events_parts = []
rows_total = 0

for chunk in src.iter_dataframes(chunksize=CHUNKSIZE, columns=cols_needed):
    # Nettoyage minimal
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    # cast date
    chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
    chunk = chunk.dropna(subset=[TIME_COL])

    # Filtre temporel (facultatif mais recommandé pour RAM)
    if KEEP_MONTHS is not None:
        cutoff = (pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS))
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    # Dédup (1 ligne par (client, mois)) : on garde la plus récente
    # Si DATMAJ est déjà mensuel, cette étape est surtout un "last duplicate win"
    chunk = (
        chunk.sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
    )

    events_parts.append(chunk)
    rows_total += len(chunk)

    # garde-fou RAM
    if MAX_ROWS_HINT is not None and rows_total >= MAX_ROWS_HINT:
        break

events_df = pd.concat(events_parts, ignore_index=True) if events_parts else pd.DataFrame(columns=cols_needed)

# (Optionnel) downcast pour soulager la RAM
for c in events_df.select_dtypes(include=["int64"]).columns:
    events_df[c] = pd.to_numeric(events_df[c], downcast="integer")

print(f"   ✅ events_df ready: {events_df.shape} | cols={list(events_df.columns)}")

# -----------------------------------------
# 3) FS légère sur profil (toujours facult.)
#    — ici on peut s’en passer; si tu veux:
# -----------------------------------------
try:
    fs_cfg = FeatureSelectorConfig(
        sample_size=50_000, total_feature_cap=20,
        top_k_sequence=8, top_k_categorical=6,
        compute_model_importance=True, rf_n_estimators=100,
        corr_threshold=0.85, chunk_size=20_000,
        downcast_dtypes=True, correlation_batch_size=100,
        verbose=True, progress=True,
    )
    fs = FeatureSelector(fs_cfg)
    fs.fit(DATASET_MAIN, None, PRODUCT_COL)
    selected = fs.get_selected_features()
    profile_categorical_cols = selected.get("categorical_cols", [])[:6]
    profile_sequence_cols    = selected.get("sequence_cols", [])[:4]
except Exception as e:
    print(f"FS skipped (non-blocking): {e}")
    profile_categorical_cols, profile_sequence_cols = [], []

print("   profil cat :", profile_categorical_cols)
print("   profil num :", profile_sequence_cols)

# --------------------------------------------------
# 4) Config du pipeline + monkey-patch loader events
# --------------------------------------------------
cfg = blank_config()

# données événements (ces champs sont requis par le pipeline,
# mais on va ignorer le 'events_dataset' via le patch)
cfg["data"]["events_dataset"]   = "__IGNORED__"
cfg["data"]["client_id_col"]    = CLIENT_ID_COL
cfg["data"]["event_time_col"]   = TIME_COL
cfg["data"]["product_col"]      = PRODUCT_COL
cfg["data"]["event_extra_cols"] = EXTRA_EVENT_COLS

# données profil (facultatif) : on réutilise la table principale
cfg["data"]["dataset_name"]             = DATASET_MAIN
cfg["data"]["profile_join_key"]         = CLIENT_ID_COL
cfg["data"]["profile_categorical_cols"] = profile_categorical_cols
cfg["data"]["profile_sequence_cols"]    = profile_sequence_cols

# fenêtre temporelle (doit coller à ce que tu veux exploiter)
cfg["sequence"]["months_lookback"] = 24
cfg["sequence"]["time_granularity"] = "M"
cfg["sequence"]["min_events_per_client"] = 1
cfg["sequence"]["target_horizon"] = 1
cfg["sequence"]["pad_value"] = 0
cfg["sequence"]["build_target_from_events"] = True

# cible : exclure "aucune_souscription" + regrouper rares
cfg["features"]["exclude_target_values"] = ["aucune_souscription"]
cfg["features"]["merge_rare_threshold"]  = 200
cfg["features"]["other_class_name"]      = "AUTRES_PRODUITS"

# modèle costaud ➜ réduis si OOM
cfg["model"]["d_model"]             = 512   # 768 → 512 pour RAM
cfg["model"]["n_heads"]             = 16    # 24  → 16  pour RAM
cfg["model"]["n_layers"]            = 6     # 8   → 6   pour RAM
cfg["model"]["dropout"]             = 0.10
cfg["model"]["max_sequence_length"] = 24
cfg["model"]["vocab_size"]          = 2000

# entraînement (mini-batch + pondération classes)
cfg["training"]["batch_size"]      = 64     # 64 ou 32 si OOM
cfg["training"]["num_epochs"]      = 15
cfg["training"]["learning_rate"]   = 5e-4
cfg["training"]["weight_decay"]    = 1e-4
cfg["training"]["val_split"]       = 0.20
cfg["training"]["class_weighting"] = True
cfg["training"]["gradient_clip"]   = 1.0
cfg["training"]["optimizer"]       = "adamw"

# sorties : on ne sauvegarde rien en datasets (tout en mémoire)
cfg["outputs"]["features_dataset"]        = None
cfg["outputs"]["predictions_dataset"]     = None
cfg["outputs"]["metrics_dataset"]         = None
cfg["outputs"]["model_artifacts_dataset"] = None

cfg["runtime"]["verbose"]  = True
cfg["runtime"]["progress"] = True
cfg["runtime"]["seed"]     = 42

print("   ARCHI:", f"{cfg['model']['n_layers']}L-{cfg['model']['n_heads']}H-{cfg['model']['d_model']}D")

# ➜ Monkey-patch : on remplace le chargeur d’événements du pipeline
def _load_events_df_override(cfg_in):
    # on renvoie NOTRE DataFrame en mémoire plutôt que de lire un dataset
    return events_df

pc._load_events_df = _load_events_df_override  # ✅ patch

# -----------------------------------------
# 5) Lancer l’entraînement (tout en mémoire)
# -----------------------------------------
results = run_training(cfg)
print("== TRAIN DONE ==")

# -----------------------------------------
# 6) Afficher un résumé rapide
# -----------------------------------------
m = results.get("metrics", {})
print("metrics:", {k: round(float(v), 4) for k, v in m.items() if v is not None})
print("data_info:", results.get("data_info", {}))
print("model_info:", results.get("model_info", {}))
