#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, logging, traceback
from datetime import datetime
import pandas as pd
import numpy as np

import dataiku
from t4rec_toolkit.pipeline_core import blank_config, run_training

# =========================
# PARAMS — SMOKE TEMPORAL
# =========================
DATASET_MAIN      = "BASE_SCORE_COMPLETE_prepared"
CLIENT_ID_COL     = "NUMTECPRS"
TIME_COL          = "DATMAJ"
PRODUCT_COL       = "SOUSCRIPTION_PRODUIT_1M"

LIMIT_ROWS        = 1_000       # augmente progressivement (2000, 5000, 10000…)
CHUNKSIZE         = 50_000

KEEP_MONTHS_INIT  = 12          # fenêtre d’historique de base
MIN_CLIENT_MONTHS_INIT = 2      # >= 2 mois pour garder l’aspect temporel
TARGET_EXCLUDE    = ["Aucune_Proposition"]

# Modèle / training (CPU-friendly)
D_MODEL    = 128
N_HEADS    = 2
N_LAYERS   = 2
BATCH_SIZE = 8
EPOCHS     = 2
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.20
MERGE_RARE_THRESHOLD = 1000

# =========================
# LOGGING
# =========================
os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("driver")

# =========================
# BUILD ONE ATTEMPT
# =========================
def _build_once(keep_months, min_client_months, exclude_on_events):
    src = dataiku.Dataset(DATASET_MAIN)

    # colonnes
    try:
        schema_cols = list(src.get_dataframe(limit=0).columns)
    except Exception:
        cfg = src.get_config()
        schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

    need = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
    missing = [c for c in need if c not in schema_cols]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {DATASET_MAIN}: {missing}")

    rows_total = 0
    chunks = []
    log.info(f"[attempt] iter (keep_months={keep_months}, min_client_months={min_client_months}, exclude_on_events={exclude_on_events})")
    for chunk in src.iter_dataframes(chunksize=CHUNKSIZE, parse_dates=False, infer_with_pandas=True):
        inter = [c for c in chunk.columns if c in need]
        if not inter:
            continue
        chunk = chunk[inter].dropna(subset=[CLIENT_ID_COL, TIME_COL])

        dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
        chunk[TIME_COL] = dt
        chunk = chunk.dropna(subset=[TIME_COL])

        # filtre temporel (None => désactivé)
        if keep_months is not None:
            cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=int(keep_months))
            chunk = chunk[chunk[TIME_COL] >= cutoff]

        # exclusion côté événements (si demandé)
        if exclude_on_events and TARGET_EXCLUDE:
            excl = [s.strip() for s in TARGET_EXCLUDE]
            chunk = chunk[~chunk[PRODUCT_COL].astype(str).str.strip().isin(excl)]

        if chunk.empty:
            continue

        # dédup (client, date)
        chunk = (
            chunk.sort_values([CLIENT_ID_COL, TIME_COL])
                 .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
        )
        if chunk.empty:
            continue

        # respect du LIMIT_ROWS
        remaining = LIMIT_ROWS - rows_total
        if remaining <= 0:
            log.info(f"STOP: limite {LIMIT_ROWS:,} atteinte.")
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        chunks.append(chunk)
        rows_total += len(chunk)
        log.info(f"   + {len(chunk):,} rows (cum={rows_total:,})")
        if rows_total >= LIMIT_ROWS:
            log.info(f"STOP: limite {LIMIT_ROWS:,} atteinte.")
            break

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=need)

    # garder clients avec >= min_client_months mois
    if not df.empty and min_client_months and min_client_months > 1:
        months_per_client = (
            df.assign(_month=df[TIME_COL].dt.to_period("M"))
              .groupby(CLIENT_ID_COL)["_month"].nunique()
        )
        keep_ids = set(months_per_client[months_per_client >= min_client_months].index)
        before = len(df)
        df = df[df[CLIENT_ID_COL].isin(keep_ids)]
        log.info(f"Filtre MIN_CLIENT_MONTHS>={min_client_months}: {before:,}→{len(df):,} lignes; clients={len(keep_ids):,}")

    return df

# =========================
# STEP 1 — BUILD with FALLBACKS
# =========================
def build_df_events_with_fallbacks():
    log.info("== STEP 1: BUILD EVENTS (with fallbacks) ==")

    km = KEEP_MONTHS_INIT
    mm = MIN_CLIENT_MONTHS_INIT

    attempts = [
        (km,       mm, True),   # nominal
        (km*2,     mm, True),   # élargir fenêtre
        (km*3,     mm, True),
        (km*2,     1,  True),   # assouplir contrainte client
        (km*3,     1,  True),
        (km*3,     1,  False),  # dernier recours: pas d'exclusion côté événements
        (None,     1,  False),  # secours absolu: toute la période dispo
    ]

    for i, (keep_m, min_m, excl_ev) in enumerate(attempts, 1):
        log.info(f"--- Attempt {i}/{len(attempts)} ---")
        df = _build_once(keep_m, min_m, excl_ev)
        log.info(f"Attempt {i} result: shape={df.shape}")
        if not df.empty:
            # petit récap
            try:
                log.info(df.head(8).to_string(index=False))
                log.info(
                    "Stats — rows=%s | clients uniques=%s | mois distincts=%s",
                    f"{len(df):,}",
                    f"{df[CLIENT_ID_COL].nunique():,}",
                    df[TIME_COL].dt.to_period("M").nunique(),
                )
                log.info("Top target values:\n" + df[PRODUCT_COL].astype(str).value_counts().head(10).to_string())
            except Exception:
                pass
            # on retourne le df ET l’info si on a exclu côté events
            return df, excl_ev

    raise RuntimeError("Impossible d'obtenir des événements non vides même après fallbacks (élargir LIMIT_ROWS ou vérifier les données).")

# =========================
# STEP 2 — CONFIG
# =========================
def make_config(df_events, excluded_on_events: bool) -> dict:
    log.info("== STEP 2: CONFIG (in-memory, no IO) ==")
    cfg = blank_config()

    # données (in-memory)
    cfg["data"]["events_df"]        = df_events
    cfg["data"]["client_id_col"]    = CLIENT_ID_COL
    cfg["data"]["event_time_col"]   = TIME_COL
    cfg["data"]["product_col"]      = PRODUCT_COL
    cfg["data"]["event_extra_cols"] = []   # pas de profil pour limiter la RAM

    # séquence
    # on met months_lookback à min(KEEP_MONTHS_INIT, nb_mois dispo)
    n_months_dispo = int(df_events[TIME_COL].dt.to_period("M").nunique())
    eff_lookback = min(KEEP_MONTHS_INIT, max(1, n_months_dispo))
    cfg["sequence"]["months_lookback"]          = eff_lookback
    cfg["sequence"]["time_granularity"]         = "M"
    cfg["sequence"]["min_events_per_client"]    = 1
    cfg["sequence"]["target_horizon"]           = 1
    cfg["sequence"]["pad_value"]                = 0
    cfg["sequence"]["build_target_from_events"] = True

    # cible — si on n’a PAS pu exclure côté événements,
    # on sécurise l’exclusion côté y (dans run_training)
    cfg["features"]["exclude_target_values"] = TARGET_EXCLUDE if not excluded_on_events else TARGET_EXCLUDE
    cfg["features"]["merge_rare_threshold"]  = MERGE_RARE_THRESHOLD
    cfg["features"]["other_class_name"]      = "AUTRES_PRODUITS"

    # modèle
    cfg["model"]["d_model"]             = D_MODEL
    cfg["model"]["n_heads"]             = N_HEADS
    cfg["model"]["n_layers"]            = N_LAYERS
    cfg["model"]["dropout"]             = 0.10
    cfg["model"]["max_sequence_length"] = eff_lookback
    cfg["model"]["vocab_size"]          = 2000

    # training
    cfg["training"]["batch_size"]      = BATCH_SIZE
    cfg["training"]["num_epochs"]      = EPOCHS
    cfg["training"]["learning_rate"]   = LEARNING_RATE
    cfg["training"]["weight_decay"]    = 1e-4
    cfg["training"]["val_split"]       = VAL_SPLIT
    cfg["training"]["class_weighting"] = True
    cfg["training"]["gradient_clip"]   = 1.0
    cfg["training"]["optimizer"]       = "adamw"

    # sorties désactivées (smoke)
    cfg["outputs"]["features_dataset"]        = None
    cfg["outputs"]["predictions_dataset"]     = None
    cfg["outputs"]["metrics_dataset"]         = None
    cfg["outputs"]["model_artifacts_dataset"] = None
    cfg["outputs"]["local_dir"]               = "output"

    # runtime
    cfg["runtime"]["verbose"]  = True
    cfg["runtime"]["progress"] = True
    cfg["runtime"]["seed"]     = 42

    log.info(f"Archi: {cfg['model']['n_layers']}L-{cfg['model']['n_heads']}H-{cfg['model']['d_model']}D")
    log.info(f"Seq:   {cfg['sequence']['months_lookback']} mois | horizon={cfg['sequence']['target_horizon']}")
    log.info(f"Exclu (events level)?: {excluded_on_events}")
    log.info("OK config.")
    return cfg

# =========================
# TRAIN & SUMMARY
# =========================
def train(cfg):
    log.info("== STEP 3: TRAIN (CPU) ==")
    t0 = time.time()
    try:
        results = run_training(cfg)
        log.info("Training finished.")
    except Exception as e:
        log.error(f"ERREUR TRAIN: {e}\n{traceback.format_exc()}")
        raise
    log.info(f"Train time: {time.time()-t0:.1f}s")
    return results

def print_summary(results):
    m = results.get("metrics", {})
    di = results.get("data_info", {})
    mi = results.get("model_info", {})

    for k in ["accuracy", "precision", "recall", "f1"]:
        if k in m:
            log.info(f"{k:9s} : {m[k]:.4f}")

    log.info(f"Model: {mi.get('architecture','N/A')} | params≈ {mi.get('total_params','NA'):,}")
    log.info(f"Data : clients={di.get('n_clients','NA')} | seq_len={di.get('seq_len','NA')} | classes={di.get('n_classes','NA')}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    log.info("=== DRIVER START (in-memory, with fallbacks) ===")
    try:
        df_events, excluded_on_events = build_df_events_with_fallbacks()
    except Exception as e:
        log.error(str(e))
        sys.exit(1)

    cfg = make_config(df_events, excluded_on_events)
    results = train(cfg)
    print_summary(results)
    log.info("=== DRIVER DONE ===")

