# -*- coding: utf-8 -*-
"""
Driver 'bounded events' + (optionnel) mini-train CPU
- Lit le dataset source en chunks
- Filtre sur N derniers mois, exclut des labels, limite à LIMIT_ROWS
- Garde-fous pour éviter un df_events vide (relaxation progressive)
- Ecrit df_event + results
- Optionnel: lance un petit entraînement CPU via pipeline_core.run_training
"""

import os, sys, time, logging
from datetime import datetime
import pandas as pd
import numpy as np

# ==== Dataiku ====
import dataiku

# ==== Ton toolkit ====
from t4rec_toolkit.pipeline_core import blank_config, run_training

# ---------------- CONFIG ----------------
DATASET_MAIN        = "BASE_SCORE_COMPLETE_prepared"   # dataset source
OUT_EVENTS_NAME     = "df_event"                       # dataset de sortie events (crée-le dans le Flow)
OUT_RESULTS_NAME    = "results"                        # dataset de logs (crée-le dans le Flow)

CLIENT_ID_COL       = "NUMTECPRS"
TIME_COL            = "DATMAJ"                         # fin de mois
PRODUCT_COL         = "SOUSCRIPTION_PRODUIT_1M"

EXCLUDE_LABELS      = ("Aucune_Proposition",)          # labels à exclure
KEEP_MONTHS         = 12                                # fenêtre temporelle (None = no filter)
MIN_CLIENT_MONTHS   = 1                                 # nb min de mois par client (1 = pas de contrainte)
CHUNKSIZE           = 200_000                           # chunk lecture
LIMIT_ROWS          = 10_000                            # STRICT hard-stop sur les lignes retenues

# Entraînement (toggle)
TRAIN_MODEL         = True                              # False pour juste tester le build
LOOKBACK_FOR_MODEL  = 6                                 # <= nb de mois retenus réels
D_MODEL             = 128
N_HEADS             = 2
N_LAYERS            = 2
BATCH_SIZE          = 8
EPOCHS              = 2
MERGE_RARE_THRESHOLD= 1000
VAL_SPLIT           = 0.2
LEARNING_RATE       = 5e-4

# -----------------------------------------

# ==== Logging ====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("driver")

def month_cutoff(n_months: int):
    if n_months is None:
        return None
    now = pd.Timestamp.now().normalize()
    # DATMAJ est fin de mois → on prend la même granularité (période M)
    return (now - pd.DateOffset(months=n_months)).to_period("M").to_timestamp(how="end")

def build_events_bounded(
    ds_name: str,
    limit_rows: int,
    keep_months: int | None,
    min_client_months: int,
    exclude_labels: tuple[str, ...],
    chunksize: int,
) -> pd.DataFrame:
    """
    Lecture chunkée + filtres + hard stop. Pas de param 'partitions'.
    Garde-fous: si vide, on relaxe automatiquement.
    """
    src = dataiku.Dataset(ds_name)

    # 1) découverte colonnes (sans API non-portable)
    try:
        schema_cols = list(src.get_dataframe(limit=0).columns)
    except Exception:
        cfg = src.get_config()
        schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

    cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
    missing = [c for c in cols_needed if c not in schema_cols]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {ds_name}: {missing}")

    # 2) itération
    cutoff = month_cutoff(keep_months)
    rows_total = 0
    chunks = []

    log.info(f"[STEP1] limit_rows={limit_rows:,} | keep_months={keep_months} | exclude={exclude_labels}")

    for chunk in src.iter_dataframes(chunksize=chunksize, parse_dates=False, infer_with_pandas=True):
        # garde uniquement le nécessaire
        inter = [c for c in chunk.columns if c in cols_needed]
        if not inter:
            continue
        chunk = chunk[inter].dropna(subset=[CLIENT_ID_COL, TIME_COL])

        # parse DATMAJ → fin de mois tz-naive
        dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=False)
        chunk = chunk.assign(**{TIME_COL: dt}).dropna(subset=[TIME_COL])

        # normalise à fin de mois (au cas où)
        chunk[TIME_COL] = chunk[TIME_COL].dt.to_period("M").dt.to_timestamp(how="end")

        # filtre temporel
        if cutoff is not None:
            chunk = chunk[chunk[TIME_COL] >= cutoff]

        # exclure labels indésirables (au niveau events)
        if exclude_labels:
            chunk = chunk[~chunk[PRODUCT_COL].astype(str).str.strip().isin(exclude_labels)]

        if chunk.empty:
            continue

        # dédup (client, date) → garder la + récente (au cas où)
        chunk = (
            chunk.sort_values([CLIENT_ID_COL, TIME_COL])
                 .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
        )

        # tronquage pour respecter la limite stricte
        remaining = limit_rows - rows_total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        chunks.append(chunk)
        rows_total += len(chunk)
        log.info(f"   + {len(chunk):,} rows (cum={rows_total:,})")
        if rows_total >= limit_rows:
            log.info("STOP: limit %s reached.", f"{limit_rows:,}")
            break

    df_events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_needed)

    # 3) filtre clients avec au moins X mois (si T>1)
    if min_client_months > 1 and not df_events.empty:
        cnt = df_events.groupby(CLIENT_ID_COL)[TIME_COL].nunique()
        keep_clients = set(cnt[cnt >= min_client_months].index)
        before = len(df_events)
        df_events = df_events[df_events[CLIENT_ID_COL].isin(keep_clients)]
        log.info(f"Filtre MIN_CLIENT_MONTHS>={min_client_months}: {before:,}→{len(df_events):,} lignes; clients={len(keep_clients):,}")

    return df_events

def ensure_non_empty_events() -> pd.DataFrame:
    """
    Essaie plusieurs relaxations si vide.
    """
    tries = [
        dict(keep_months=KEEP_MONTHS, min_client_months=MIN_CLIENT_MONTHS, exclude_labels=EXCLUDE_LABELS),
        dict(keep_months=KEEP_MONTHS, min_client_months=1,                  exclude_labels=EXCLUDE_LABELS),
        dict(keep_months=None,        min_client_months=1,                  exclude_labels=EXCLUDE_LABELS),
        dict(keep_months=None,        min_client_months=1,                  exclude_labels=tuple()),
    ]
    for i, t in enumerate(tries, 1):
        log.info(f"--- Attempt {i}/{len(tries)} ---")
        df_ev = build_events_bounded(
            ds_name=DATASET_MAIN,
            limit_rows=LIMIT_ROWS,
            keep_months=t["keep_months"],
            min_client_months=t["min_client_months"],
            exclude_labels=t["exclude_labels"],
            chunksize=CHUNKSIZE,
        )
        log.info(f"Attempt {i} result: shape={df_ev.shape}")
        if not df_ev.empty:
            return df_ev
    # si toujours vide → on prend un micro-sample brut (aucun filtre)
    log.warning("Tous les essais vides. On prend un micro-sample brut (1000 lignes) sans filtre.")
    src = dataiku.Dataset(DATASET_MAIN)
    df = src.get_dataframe(limit=1_000)[[CLIENT_ID_COL, TIME_COL, PRODUCT_COL]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce").dt.to_period("M").dt.to_timestamp(how="end")
    df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    return df

def write_dataset_safe(ds_name: str, df: pd.DataFrame):
    ds = dataiku.Dataset(ds_name)
    ds.write_with_schema(df)

def main():
    log.info("=== DRIVER START (bounded events) ===")

    # 1) BUILD EVENTS (avec garde-fous)
    df_events = ensure_non_empty_events()

    # 1.b) Stats rapides
    if df_events.empty:
        raise RuntimeError("df_events est vide malgré les fallbacks.")
    log.info("\n%s", df_events.head(8))
    n_clients = df_events[CLIENT_ID_COL].nunique()
    n_months  = df_events[TIME_COL].dt.to_period("M").nunique()
    log.info("Stats — rows=%s | clients=%s | months=%s", f"{len(df_events):,}", f"{n_clients:,}", f"{n_months}")

    vc = df_events[PRODUCT_COL].astype(str).value_counts()
    log.info("Top target values:\n%s", vc.head(10))

    # 2) WRITE df_event (sortie n°1)
    write_dataset_safe(OUT_EVENTS_NAME, df_events)
    log.info("Wrote %s: %s", OUT_EVENTS_NAME, df_events.shape)

    # 3) (OPTIONNEL) TRAIN
    results_row = {
        "run_id":            f"run_{int(time.time())}",
        "ts":                pd.Timestamp.now(),
        "rows_scanned":      int(len(df_events)),
        "rows_kept":         int(len(df_events)),
        "keep_months":       int(KEEP_MONTHS) if KEEP_MONTHS is not None else None,
        "min_client_months": int(MIN_CLIENT_MONTHS),
        "limit_rows":        int(LIMIT_ROWS),
        "chunksize":         int(CHUNKSIZE),
        "time_budget_secs":  None,
        "n_chunks":          None,
        "seq_lookback":      int(min(LOOKBACK_FOR_MODEL, int(n_months) if n_months else LOOKBACK_FOR_MODEL)),
        "d_model":           int(D_MODEL),
        "n_heads":           int(N_HEADS),
        "n_layers":          int(N_LAYERS),
        "batch_size":        int(BATCH_SIZE),
        "merge_rare_threshold": int(MERGE_RARE_THRESHOLD),
        "exclude_labels":    ",".join(EXCLUDE_LABELS) if EXCLUDE_LABELS else "",
        "n_classes":         None,
        "val_accuracy":      None,
        "precision":         None,
        "recall":            None,
        "f1":                None,
        "train_seconds":     None,
        "notes":             "bounded events",
    }

    if TRAIN_MODEL:
        log.info("== TRAIN: minimal CPU config ==")
        t0 = time.time()
        cfg = blank_config()
        # alimentation en mémoire pour éviter des I/O
        cfg["data"]["events_df"]            = df_events[[CLIENT_ID_COL, TIME_COL, PRODUCT_COL]].copy()
        cfg["data"]["events_dataset"]       = ""  # non utilisé
        cfg["data"]["client_id_col"]        = CLIENT_ID_COL
        cfg["data"]["event_time_col"]       = TIME_COL
        cfg["data"]["product_col"]          = PRODUCT_COL
        cfg["data"]["event_extra_cols"]     = []  # on n'utilise pas le profil pour limiter la RAM

        cfg["sequence"]["months_lookback"]  = results_row["seq_lookback"]
        cfg["sequence"]["time_granularity"] = "M"
        cfg["sequence"]["min_events_per_client"] = 1
        cfg["sequence"]["target_horizon"]   = 1
        cfg["sequence"]["pad_value"]        = 0
        cfg["sequence"]["build_target_from_events"] = True

        cfg["features"]["exclude_target_values"] = list(EXCLUDE_LABELS)
        cfg["features"]["merge_rare_threshold"]  = MERGE_RARE_THRESHOLD
        cfg["features"]["other_class_name"]      = "AUTRES_PRODUITS"

        cfg["model"]["d_model"]            = D_MODEL
        cfg["model"]["n_heads"]            = N_HEADS
        cfg["model"]["n_layers"]           = N_LAYERS
        cfg["model"]["dropout"]            = 0.10
        cfg["model"]["max_sequence_length"] = results_row["seq_lookback"]
        cfg["model"]["vocab_size"]         = 2000

        cfg["training"]["batch_size"]      = BATCH_SIZE
        cfg["training"]["num_epochs"]      = EPOCHS
        cfg["training"]["learning_rate"]   = LEARNING_RATE
        cfg["training"]["weight_decay"]    = 1e-4
        cfg["training"]["val_split"]       = VAL_SPLIT
        cfg["training"]["class_weighting"] = True
        cfg["training"]["gradient_clip"]   = 1.0
        cfg["training"]["optimizer"]       = "adamw"

        cfg["outputs"]["features_dataset"]        = None
        cfg["outputs"]["predictions_dataset"]     = None
        cfg["outputs"]["metrics_dataset"]         = None
        cfg["outputs"]["model_artifacts_dataset"] = None
        cfg["outputs"]["local_dir"]               = "output"

        cfg["runtime"]["verbose"]  = True
        cfg["runtime"]["progress"] = True
        cfg["runtime"]["seed"]     = 42

        # Entraînement
        res = run_training(cfg)

        train_secs = time.time() - t0
        m = res.get("metrics", {})
        results_row.update({
            "n_classes":   res.get("data_info", {}).get("n_classes"),
            "val_accuracy": float(m.get("accuracy")) if m else None,
            "precision":   float(m.get("precision")) if m else None,
            "recall":      float(m.get("recall")) if m else None,
            "f1":          float(m.get("f1")) if m else None,
            "train_seconds": float(train_secs),
            "notes":       "bounded events + mini-train CPU",
        })
        log.info("Training done: acc=%.4f | f1=%.4f | n_classes=%s",
                 results_row["val_accuracy"] or -1,
                 results_row["f1"] or -1,
                 results_row["n_classes"])

    # 4) WRITE results (sortie n°2) — schéma SANS doublons
    RESULTS_COLS = [
        "run_id","ts","rows_scanned","rows_kept","keep_months","min_client_months",
        "limit_rows","chunksize","time_budget_secs","n_chunks","seq_lookback",
        "d_model","n_heads","n_layers","batch_size","merge_rare_threshold",
        "exclude_labels","n_classes","val_accuracy","precision","recall","f1",
        "train_seconds","notes"
    ]
    # normalise ordre + types
    row_norm = {k: results_row.get(k, None) for k in RESULTS_COLS}
    if row_norm["ts"] is not None:
        row_norm["ts"] = pd.to_datetime(row_norm["ts"])
    results_df = pd.DataFrame([row_norm], columns=RESULTS_COLS)

    write_dataset_safe(OUT_RESULTS_NAME, results_df)
    log.info("Wrote %s: %s", OUT_RESULTS_NAME, results_df.shape)

    log.info("=== DRIVER DONE ===")

if __name__ == "__main__":
    main()



