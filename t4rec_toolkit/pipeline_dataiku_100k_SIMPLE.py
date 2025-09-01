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
# 2) FS (profil) => SAVE EVENTS => CONFIG => TRAIN => METRICS (LOGGÉ À FOND)
# ===============================================================

import os, sys, time, logging
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import dataiku

# -- Console non-bufferisée pour voir les prints en live
os.environ["PYTHONUNBUFFERED"] = "1"

# -- Logger root au niveau INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)
    logger.info(msg)

start_all = time.time()
log("=== DRIVER START ===")

# --- Toolkit (ta lib) ---
from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig
from t4rec_toolkit.pipeline_core import (
    blank_config,
    run_training,
    evaluate_topk_metrics_nbo,   # pour fallback d'affichage Top-K
)

# ------------ PARAMS ------------
DATASET_MAIN  = "BASE_SCORE_COMPLETE_prepared"   # table profil
EVENTS_TMP    = "df_event"                       # dataset Dataiku pour events
CLIENT_ID_COL = "NUMTECPRS"
TIME_COL      = "DATMAJ"
PRODUCT_COL   = "SOUSCRIPTION_PRODUIT_1M"

# Profil (fallback si FS indispo / vide)
PROFILE_CATEGORICAL_CANDIDATES = ["MEILLEURE_CARTE_DETENUE_M", "LIBFAMCSP", "CONNAISSANCE_MIF"]
PROFILE_SEQUENCE_CANDIDATES    = ["AGE"]

# Cible
TARGET_EXCLUDE = ["aucune_souscription"]

# Fenêtre temporelle
MONTHS_LOOKBACK = 24

# Modèle & entraînement (paramètres anti-OOM)
D_MODEL    = 256    # 256/384/768 selon RAM
N_HEADS    = 4      # 4/8/16/24
N_LAYERS   = 2      # 2..8
BATCH_SIZE = 16     # 16/32/64
EPOCHS     = 10     # augmenter ensuite
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.20

# --------------------------------------------------------------
# Sanity check df_events
# --------------------------------------------------------------
log("== STEP 0: CHECK df_events en mémoire ==")
t0 = time.time()

if "df_events" not in globals():
    raise RuntimeError("df_events manquant : exécute d'abord la cellule de construction des événements.")

assert isinstance(df_events, pd.DataFrame), "df_events doit être un DataFrame pandas"
need_cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
missing = [c for c in need_cols if c not in df_events.columns]
if missing:
    raise ValueError(f"df_events ne contient pas les colonnes requises: {missing}")

log(f"df_events shape: {df_events.shape}")
log(f"df_events dtypes:\n{df_events.dtypes}")

# aperçu léger
log("Aperçu df_events.head():")
print(df_events.head(5), flush=True)

# stats rapide
try:
    n_clients = df_events[CLIENT_ID_COL].nunique()
    n_rows = len(df_events)
    n_months = df_events[TIME_COL].dt.to_period("M").nunique() if np.issubdtype(df_events[TIME_COL].dtype, np.datetime64) else "NA"
    log(f"Stats rapides — rows={n_rows:,} | clients uniques={n_clients:,} | mois distincts={n_months}")
except Exception as e:
    log(f"Stats rapides impossibles (non bloquant): {e}")

# check target exclude
try:
    vc = df_events[PRODUCT_COL].astype(str).value_counts(dropna=False)
    log(f"TOP valeurs de la cible (head 10):\n{vc.head(10)}")
    excl_present = [x for x in TARGET_EXCLUDE if x in vc.index]
    log(f"Présence des valeurs à exclure dans df_events: {excl_present}")
except Exception as e:
    log(f"Value counts cible impossible (non bloquant): {e}")

log(f"STEP 0 DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
# FS (profil)
# --------------------------------------------------------------
log("== STEP A: FS (profil) ==")
t0 = time.time()
seq_cols_fs, cat_cols_fs = [], []
try:
    fs_cfg = FeatureSelectorConfig(
        sample_size=50_000, total_feature_cap=20,
        top_k_sequence=10, top_k_categorical=8,
        compute_model_importance=True, rf_n_estimators=100,
        corr_threshold=0.85, chunk_size=20_000,
        downcast_dtypes=True, correlation_batch_size=100,
        gc_frequency=50, report_dataset=None, verbose=True, progress=True,
    )
    fs = FeatureSelector(fs_cfg)
    log("FS: .fit() sur DATASET_MAIN...")
    fs.fit(DATASET_MAIN, None, PRODUCT_COL)
    selected = fs.get_selected_features()
    seq_cols_fs = selected.get("sequence_cols", [])
    cat_cols_fs = selected.get("categorical_cols", [])
    log(f"FS OK. seq_cols_fs={seq_cols_fs} | cat_cols_fs={cat_cols_fs}")
except Exception as e:
    log(f"FS facultative – on continue sans: {e}\n{traceback.format_exc()}")

# fallback si FS vide
profile_categorical_cols = [c for c in PROFILE_CATEGORICAL_CANDIDATES if (not cat_cols_fs) or (c in cat_cols_fs)]
profile_sequence_cols    = [c for c in PROFILE_SEQUENCE_CANDIDATES    if (not seq_cols_fs) or (c in seq_cols_fs)]
log(f"Profil retenu — cat: {profile_categorical_cols}")
log(f"Profil retenu — num: {profile_sequence_cols}")
log(f"STEP A DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
# SAVE df_events -> Dataiku dataset
# --------------------------------------------------------------
log("== STEP B: SAVE df_events -> Dataiku dataset ==")
t0 = time.time()

# force l'ordre core (client, date, produit)
base_cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
extras = [c for c in df_events.columns if c not in base_cols]
df_events = df_events[base_cols + extras]

try:
    # Si le dataset n'existe pas dans le Flow, crée-le d'abord dans l'UI (tu l'as déjà fait).
    out_ds = dataiku.Dataset(EVENTS_TMP)
    out_ds.write_with_schema(df_events)
    log(f"Saved events -> {EVENTS_TMP} : {df_events.shape}")
except Exception as e:
    log(f"Écriture {EVENTS_TMP} échouée: {e}\n{traceback.format_exc()}")
    raise

log(f"STEP B DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
log("== STEP C: CONFIG ==")
t0 = time.time()
config = blank_config()

# Données (événements obligatoires)
config["data"]["events_dataset"]   = EVENTS_TMP
config["data"]["client_id_col"]    = CLIENT_ID_COL
config["data"]["event_time_col"]   = TIME_COL
config["data"]["product_col"]      = PRODUCT_COL
config["data"]["event_extra_cols"] = []  # ex: ["CANAL"]

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

# Sorties
config["outputs"]["features_dataset"]        = "T4REC_FEATURES"
config["outputs"]["predictions_dataset"]     = "T4REC_PREDICTIONS"
config["outputs"]["metrics_dataset"]         = "T4REC_METRICS"
config["outputs"]["model_artifacts_dataset"] = "T4REC_MODEL"
config["outputs"]["local_dir"]               = "output"

# Runtime
config["runtime"]["verbose"]  = True
config["runtime"]["progress"] = True
config["runtime"]["seed"]     = 42

log(f"Archi: {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D")
log(f"Seq:   {config['sequence']['months_lookback']} mois | horizon={config['sequence']['target_horizon']}")
log(f"Profil: cat={config['data']['profile_categorical_cols']} | num={config['data']['profile_sequence_cols']}")
log(f"Exclu:  {config['features']['exclude_target_values']}")
log("OK.")
log(f"STEP C DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------
log("== STEP D: TRAIN ==")
t0 = time.time()
try:
    results = run_training(config)
    log("Train OK.")
except Exception as e:
    log(f"ERREUR TRAIN: {e}\n{traceback.format_exc()}")
    raise
log(f"STEP D DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
# METRICS
# --------------------------------------------------------------
log("== STEP E: METRICS ==")
t0 = time.time()

# Métriques standard
m = results.get("metrics", {})
for k in ["accuracy", "precision", "recall", "f1"]:
    if k in m:
        log(f"{k:9s} : {m[k]:.4f}")

# Info modèle / données
mi = results.get("model_info", {})
di = results.get("data_info", {})
log(f"Model: {mi.get('architecture','N/A')} | params≈ {mi.get('total_params','NA'):,}")
log(f"Data : clients={di.get('n_clients','NA')} | seq_len={di.get('seq_len','NA')} | classes={di.get('n_classes','NA')}")

# Top-K (si dataset écrit par pipeline)
saved = results.get("saved_datasets", {})
if "metrics" in saved:
    try:
        met_df = dataiku.Dataset(config["outputs"]["metrics_dataset"]).get_dataframe()
        topk_view = met_df[met_df["metric_type"]=="topk_nbo"].sort_values(["k_value","metric_name"])
        log("Top-K (échantillon depuis dataset):")
        print(topk_view.head(20), flush=True)
    except Exception as e:
        log(f"Lecture métriques dataset impossible (non bloquant): {e}")
else:
    # Fallback: calcule un Top-K rapide depuis 'results'
    preds = results.get("predictions", {})
    raw = preds.get("raw_outputs", None)
    y   = preds.get("true_classes", None)
    if raw is not None and y is not None:
        log("Top-K rapide (calcul local sur indices)…")
        inv_map = {i: f"class_{i}" for i in range(raw.shape[1])}
        tk, _ = evaluate_topk_metrics_nbo(
            predictions=raw, targets=y, inverse_target_mapping=inv_map, k_values=[1,3,5]
        )
        for k,v in tk.items():
            log(f"K={k}: P@K={v['Precision@K']:.4f} | R@K={v['Recall@K']:.4f} | F1@K={v['F1@K']:.4f} | NDCG@K={v['NDCG@K']:.4f}")

log(f"STEP E DONE in {time.time()-t0:.1f}s")

# --------------------------------------------------------------
total_s = time.time() - start_all
log(f"=== DRIVER DONE in {total_s:.1f}s ===")
