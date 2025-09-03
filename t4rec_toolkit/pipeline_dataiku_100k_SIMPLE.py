

"""
DRIVER T4REC — CPU-friendly, in-memory, avec fallbacks anti-dataset vide.

Ce script :
  1) Construit df_events à partir du dataset Dataiku "BASE_SCORE_COMPLETE_prepared"
     en lecture chunkée avec filtres progressifs (fallbacks).
  2) Exclut "Aucune_Proposition" AU NIVEAU ÉVÉNEMENTS (optionnel & recommandé ici),
     pour éviter un y dégénéré.
  3) Lance run_training(config) en PASSANT df_events en mémoire (data.events_df).
  4) Append un CSV "runs_log.csv" avec hyperparams et résultats clés.

→ Objectif : faire tourner un smoke test qui exploite VRAIMENT la dimension temporelle
 … quand l’historique existe dans les données source.
"""

import os
import sys
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

# Dataiku
import dataiku

# Ton pipeline
from t4rec_toolkit.pipeline_core import run_training, blank_config

# =========================
# CONFIG DRIVER (modifiable)
# =========================

# Dataset source & colonnes
DATASET_MAIN     = "BASE_SCORE_COMPLETE_prepared"
CLIENT_ID_COL    = "NUMTECPRS"
TIME_COL         = "DATMAJ"                      # ⚠️ Doit être la vraie date d’événement. Si c'est une date de MAJ unique, remplace par la bonne colonne.
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []                            # ex: ["CANAL", "FAMILLE"]

# Lecture & bornes
CHUNKSIZE        = 50_000
LIMIT_ROWS       = 5_000                         # monte progressivement (5k → 10k → 25k…)
KEEP_MONTHS_TRY  = [12, 24, 36]                  # gardera le 1er essai qui marche
MIN_CLIENT_MONTHS_TRY = [2, 1]                   # exige ≥2 mois pour bénéficier du temporel ; sinon fallback à 1
EXCLUDE_ON_EVENTS = True                         # exclure Aucune_Proposition AVANT cible (évite y dégénéré)

# Cible rare → "AUTRES_PRODUITS"
MERGE_RARE_THRESHOLD = 50                        # évite de tout merger ; ajuste 20–100 selon volumes

# Hyperparams modèle CPU-friendly
D_MODEL    = 128
N_HEADS    = 2
N_LAYERS   = 2
BATCH_SIZE = 8
EPOCHS     = 2
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.2

# Divers
RANDOM_SEED = 42
RUNS_CSV    = "runs_log.csv"

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("driver")

# =========================
# HELPERS
# =========================

def _read_events_chunks(limit_rows:int, keep_months:int) -> pd.DataFrame:
    """
    Lit le dataset en chunks, garde uniquement les colonnes utiles,
    nettoie la date en tz-naive et filtre sur les N derniers mois.
    S’arrête exactement à limit_rows.
    """
    src = dataiku.Dataset(DATASET_MAIN)

    # Colonnes existantes (robuste)
    try:
        schema_cols = list(src.get_dataframe(limit=0).columns)
    except Exception:
        cfg = src.get_config()
        schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

    cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
    missing = [c for c in cols_needed if c not in schema_cols]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {DATASET_MAIN}: {missing}")

    keep_set = set(cols_needed)

    chunks = []
    rows_total = 0

    log.info("Starting dataframes iterator")
    for chunk in src.iter_dataframes(
        chunksize=CHUNKSIZE,
        parse_dates=False,
        infer_with_pandas=True
    ):
        inter = [c for c in chunk.columns if c in keep_set]
        if not inter:
            continue
        chunk = chunk[inter]

        # ID/time obligatoires
        chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])

        # Date → datetime tz-naive robuste
        dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
        dt = dt.dt.tz_convert(None)
        chunk[TIME_COL] = dt
        chunk = chunk.dropna(subset=[TIME_COL])

        # Filtre N derniers mois si demandé
        if keep_months is not None:
            cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=keep_months)
            chunk = chunk[chunk[TIME_COL] >= cutoff]

        if chunk.empty:
            continue

        # Dédup stricte par (client, date) -> garde la + récente
        chunk = (
            chunk.sort_values([CLIENT_ID_COL, TIME_COL])
                 .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
        )
        if chunk.empty:
            continue

        # Application exacte de LIMIT_ROWS
        remaining = limit_rows - rows_total
        if remaining <= 0:
            log.info(f"STOP: limite {limit_rows:,} atteinte.")
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        chunks.append(chunk)
        rows_total += len(chunk)
        log.info(f"   + {len(chunk):,} rows (cum={rows_total:,})")

        if rows_total >= limit_rows:
            log.info(f"STOP: limite {limit_rows:,} atteinte.")
            break

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame(columns=cols_needed)

    return df


def _filter_min_client_months(df: pd.DataFrame, min_client_months:int) -> pd.DataFrame:
    """
    Garde les (client, mois) et filtre les clients avec >= min_client_months.
    """
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_month"] = tmp[TIME_COL].dt.to_period("M")
    grp = tmp.groupby(CLIENT_ID_COL)["_month"].nunique()
    keep_ids = set(grp[grp >= min_client_months].index)
    out = tmp[tmp[CLIENT_ID_COL].isin(keep_ids)].drop(columns=["_month"])
    log.info(f"Filtre MIN_CLIENT_MONTHS>={min_client_months}: {len(df)}→{len(out)} lignes; clients={len(keep_ids)}")
    return out


def build_events_with_fallbacks(limit_rows:int) -> tuple[pd.DataFrame, dict]:
    """
    Essaie plusieurs combinaisons (keep_months, min_client_months, exclusion) pour
    garantir un df_events non vide et si possible T>=2.
    Retourne (df_events, meta).
    """
    attempts_meta = []
    # 1) EXCLUDE_ON_EVENTS tel quel
    # 2) Si rien → désactive exclusion pour débloquer
    exclude_flags = [EXCLUDE_ON_EVENTS, False] if EXCLUDE_ON_EVENTS else [False]

    try_num = 0
    for exclude_flag in exclude_flags:
        for km in KEEP_MONTHS_TRY:
            for mcm in MIN_CLIENT_MONTHS_TRY:
                try_num += 1
                log.info(f"--- Attempt {try_num}/{len(exclude_flags)*len(KEEP_MONTHS_TRY)*len(MIN_CLIENT_MONTHS_TRY)} ---")
                log.info(f"[attempt] iter (keep_months={km}, min_client_months={mcm}, exclude_on_events={exclude_flag})")

                df = _read_events_chunks(limit_rows=limit_rows, keep_months=km)
                if df.empty:
                    attempts_meta.append((km, mcm, exclude_flag, "read=empty"))
                    continue

                # Exclusion au niveau événements (optionnel)
                if exclude_flag:
                    before = len(df)
                    df = df[df[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]
                    after = len(df)
                    log.info(f"Exclusion événement 'Aucune_Proposition': {before}→{after}")

                # Filtre profondeur temporelle client
                df = _filter_min_client_months(df, mcm)

                # Check
                if df.empty:
                    attempts_meta.append((km, mcm, exclude_flag, "post-filter=empty"))
                    log.info(f"Attempt {try_num} result: shape=(0, 3)")
                    continue

                # Stats rapides
                log.info(f"Attempt {try_num} result: shape={df.shape}")
                try:
                    log.info(f"\n{df.head(8).to_string(index=False)}")
                except:
                    pass

                n_clients = df[CLIENT_ID_COL].nunique()
                n_months = df[TIME_COL].dt.to_period("M").nunique()
                log.info(f"Stats — rows={len(df):,} | clients uniques={n_clients:,} | mois distincts={n_months}")

                # Distribution cible
                try:
                    vc = df[PRODUCT_COL].astype(str).value_counts()
                    log.info("Top target values:\n" + vc.head(10).to_string())
                except:
                    pass

                return df, {
                    "keep_months": km,
                    "min_client_months": mcm,
                    "exclude_on_events": exclude_flag,
                    "n_clients": int(n_clients),
                    "n_months": int(n_months),
                }

    # Fallback ultime : si tout vide, on lit sans filtre mois + sans exclusion + min_client_months=1
    log.warning("Tous les essais ont échoué → fallback ULTIME (no exclude, no month filter, mcm=1)")
    df = _read_events_chunks(limit_rows=limit_rows, keep_months=None)
    df = _filter_min_client_months(df, 1)
    if df.empty:
        raise RuntimeError("df_events est vide malgré le fallback ultime : vérifie TIME_COL, LIMIT_ROWS, et que le dataset contient des dates parsables.")
    n_clients = df[CLIENT_ID_COL].nunique()
    n_months = df[TIME_COL].dt.to_period("M").nunique()
    return df, {
        "keep_months": None,
        "min_client_months": 1,
        "exclude_on_events": False,
        "n_clients": int(n_clients),
        "n_months": int(n_months),
    }


# =========================
# MAIN
# =========================

def main():
    start_all = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=== DRIVER START (in-memory, with fallbacks) ===")

    # ---- STEP 1: BUILD EVENTS ----
    log.info("== STEP 1: BUILD EVENTS (with fallbacks) ==")
    df_events, meta = build_events_with_fallbacks(limit_rows=LIMIT_ROWS)

    # ---- STEP 2: CONFIG ----
    log.info("== STEP 2: CONFIG (in-memory, no IO) ==")
    cfg = blank_config()

    # Données (passage in-memory → pas d’I/O Dataiku pour events)
    cfg["data"]["events_df"]          = df_events
    cfg["data"]["events_dataset"]     = ""                 # on n'utilise pas
    cfg["data"]["dataset_name"]       = ""                 # pas de profil pour smoke test RAM-light
    cfg["data"]["client_id_col"]      = CLIENT_ID_COL
    cfg["data"]["event_time_col"]     = TIME_COL
    cfg["data"]["product_col"]        = PRODUCT_COL
    cfg["data"]["event_extra_cols"]   = []                 # garder vide pour RAM

    # Séquence (si un seul mois trouvé, on laisse months_lookback=1 pour éviter du padding vide)
    months_lb = 12
    if meta.get("n_months", 1) < 2:
        months_lb = 1
    cfg["sequence"]["months_lookback"]          = months_lb
    cfg["sequence"]["time_granularity"]         = "M"
    cfg["sequence"]["min_events_per_client"]    = 1
    cfg["sequence"]["target_horizon"]           = 1
    cfg["sequence"]["pad_value"]                = 0
    cfg["sequence"]["build_target_from_events"] = True

    # Cible / features
    # IMPORTANT : comme on a potentiellement exclu "Aucune_Proposition" au niveau événements,
    # on NE la re-exclut PAS ici pour éviter des désalignements (laisser liste vide).
    cfg["features"]["exclude_target_values"] = []
    cfg["features"]["merge_rare_threshold"]  = MERGE_RARE_THRESHOLD
    cfg["features"]["other_class_name"]      = "AUTRES_PRODUITS"

    # Modèle (CPU-friendly)
    cfg["model"]["d_model"]             = D_MODEL
    cfg["model"]["n_heads"]             = N_HEADS
    cfg["model"]["n_layers"]            = N_LAYERS
    cfg["model"]["dropout"]             = 0.10
    cfg["model"]["max_sequence_length"] = months_lb
    cfg["model"]["vocab_size"]          = 2000

    # Training
    cfg["training"]["batch_size"]      = BATCH_SIZE
    cfg["training"]["num_epochs"]      = EPOCHS
    cfg["training"]["learning_rate"]   = LEARNING_RATE
    cfg["training"]["weight_decay"]    = 1e-4
    cfg["training"]["val_split"]       = VAL_SPLIT
    cfg["training"]["class_weighting"] = True
    cfg["training"]["gradient_clip"]   = 1.0
    cfg["training"]["optimizer"]       = "adamw"

    # Sorties (on ne les utilise pas pendant le smoke ; pas de Dataiku write ici)
    cfg["outputs"]["features_dataset"]        = None
    cfg["outputs"]["predictions_dataset"]     = None
    cfg["outputs"]["metrics_dataset"]         = None
    cfg["outputs"]["model_artifacts_dataset"] = None
    cfg["outputs"]["local_dir"]               = "output"

    # Runtime
    cfg["runtime"]["verbose"]  = True
    cfg["runtime"]["progress"] = True
    cfg["runtime"]["seed"]     = RANDOM_SEED

    log.info(f"Archi: {cfg['model']['n_layers']}L-{cfg['model']['n_heads']}H-{cfg['model']['d_model']}D")
    log.info(f"Seq:   {cfg['sequence']['months_lookback']} mois | horizon={cfg['sequence']['target_horizon']}")
    log.info(f"Exclu (events level)?: {meta.get('exclude_on_events')}")
    log.info("OK config.")

    # ---- STEP 3: TRAIN ----
    log.info("== STEP 3: TRAIN (CPU) ==")
    results = run_training(cfg)
    log.info("Training finished.")
    log.info(f"Train time: {results.get('execution_time', 0.0):.1f}s")

    # ---- STEP 4: MÉTRIQUES ----
    log.info("== STEP 4: METRICS ==")
    m  = results.get("metrics", {})
    mi = results.get("model_info", {})
    di = results.get("data_info", {})
    log.info(f"accuracy  : {m.get('accuracy', 0.0):.4f}")
    log.info(f"precision : {m.get('precision', 0.0):.4f}")
    log.info(f"recall    : {m.get('recall', 0.0):.4f}")
    log.info(f"f1        : {m.get('f1', 0.0):.4f}")
    log.info(f"Model: {mi.get('architecture','N/A')} | params≈ {mi.get('total_params','NA'):,}")
    log.info(f"Data : clients={di.get('n_clients','NA')} | seq_len={di.get('seq_len','NA')} | classes={di.get('n_classes','NA')}")

    # ---- STEP 5: APPEND CSV RUNS ----
    try:
        run_row = {
            "timestamp": pd.Timestamp.now(),
            # dataset sampling
            "limit_rows": LIMIT_ROWS,
            "keep_months_effective": meta.get("keep_months"),
            "min_client_months_effective": meta.get("min_client_months"),
            "exclude_on_events": meta.get("exclude_on_events"),
            "n_clients_events": meta.get("n_clients"),
            "n_months_events": meta.get("n_months"),
            # training config
            "months_lookback": cfg["sequence"]["months_lookback"],
            "d_model": cfg["model"]["d_model"],
            "n_heads": cfg["model"]["n_heads"],
            "n_layers": cfg["model"]["n_layers"],
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["num_epochs"],
            "lr": cfg["training"]["learning_rate"],
            "merge_rare_threshold": cfg["features"]["merge_rare_threshold"],
            # results
            "seq_len_effective": di.get("seq_len"),
            "n_clients_train": di.get("n_clients"),
            "n_classes": di.get("n_classes"),
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "train_time_s": results.get("execution_time"),
        }
        df_log = pd.DataFrame([run_row])
        if os.path.exists(RUNS_CSV):
            df_log.to_csv(RUNS_CSV, mode="a", header=False, index=False)
        else:
            df_log.to_csv(RUNS_CSV, index=False)
        log.info(f"Run append -> {RUNS_CSV}")
    except Exception as e:
        log.warning(f"Impossible d’écrire le CSV de run: {e}")

    log.info(f"=== DRIVER DONE in {time.time()-start_all:.1f}s ===")


if __name__ == "__main__":
    main()


