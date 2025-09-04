
import os, sys, time, json, logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# --- Dataiku (présumé dans DSS) ---
import dataiku

# --- Ton toolkit ---
from t4rec_toolkit.pipeline_core import run_training, blank_config

# ===================== CONFIG UTILISATEUR =========================
# Dataset source (profil + événements sont dans la même table ici)
DATASET_MAIN     = "BASE_SCORE_COMPLETE_prepared"

# Colonnes
CLIENT_ID_COL    = "NUMTECPRS"
TIME_COL         = "DATMAJ"
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []  # ex: ["CANAL","FAMILLE"]

# Échantillonnage / smoke
LIMIT_ROWS          = 1_000        # nombre max de lignes events à construire (smoke)
MONTHS_BACK_TARGET  = 6            # combien de mois récents on cible
PER_MONTH_CAP       = None         # None => auto (LIMIT_ROWS // MONTHS_BACK_TARGET)
MIN_CLIENT_MONTHS   = 2            # on veut >= 2 mois / client (temporal)
EXCLUDE_ON_EVENTS   = True         # exclure "Aucune_Proposition" côté événements
RANDOM_SEED         = 42

# Hyperparams (CPU-friendly)
D_MODEL    = 128
N_HEADS    = 2
N_LAYERS   = 2
BATCH_SIZE = 8
EPOCHS     = 2
LEARNING_RATE = 5e-4
VAL_SPLIT     = 0.20

# Classes rares (réduit la tête de classe)
MERGE_RARE_THRESHOLD = 1000
OTHER_CLASS_NAME     = "AUTRES_PRODUITS"

# Sorties
OUT_DIR = "output"
EXP_LOG_CSV = os.path.join(OUT_DIR, "experiment_log.csv")

# ===================== LOGGING =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("driver")


# ===================== HELPERS =========================
def _to_month_label(p: str) -> str:
    """Normalise une partition/valeur en libellé 'YYYY-MM'."""
    try:
        return pd.to_datetime(p).strftime("%Y-%m")
    except Exception:
        if isinstance(p, str) and len(p) == 7 and p[4] == "-":
            return p
        if isinstance(p, str) and len(p) in (6, 8) and p[:4].isdigit():
            if len(p) == 6:
                return f"{p[:4]}-{p[4:]}"
            if len(p) == 8:
                return f"{p[:4]}-{p[4:6]}"
        return str(p)


def _list_month_partitions(ds: dataiku.Dataset) -> List[str]:
    """Liste des mois disponibles (partitionnés DSS ou inférés via DATMAJ)."""
    # 1) Dataset partitionné ?
    try:
        parts = ds.list_partitions()
        parts = sorted(parts, reverse=True)
        if parts:
            return [_to_month_label(x) for x in parts]
    except Exception:
        pass

    # 2) Fallback: scanner DATMAJ en chunks
    months = set()
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL]
    for ch in ds.iter_dataframes(chunksize=200_000, parse_dates=False, infer_with_pandas=True, columns=cols):
        dt = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
        months.update(dt.dropna().dt.to_period("M").astype(str).unique().tolist())
        if len(months) >= 48:
            break
    return sorted(list(months), reverse=True)


def _read_month_sample(
    ds: dataiku.Dataset,
    month_label: str,
    per_month_cap: int,
    exclude_on_events: bool
) -> pd.DataFrame:
    """Lit un échantillon pour un mois donné (partitionné ou filtré sur DATMAJ)."""
    cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)

    # 1) Tentative "partitions"
    try:
        df = ds.get_dataframe(partitions=month_label, columns=cols, limit=None)
        if per_month_cap and len(df) > per_month_cap:
            df = df.sample(n=per_month_cap, random_state=RANDOM_SEED)
    except Exception:
        # 2) Fallback: filtrer DATMAJ == month_label
        df_list = []
        for ch in ds.iter_dataframes(chunksize=200_000, parse_dates=False, infer_with_pandas=True, columns=cols):
            dt = pd.to_datetime(ch[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
            ch = ch.loc[dt.dt.to_period("M").astype(str) == month_label, cols]
            if not ch.empty:
                df_list.append(ch)
            if sum(len(x) for x in df_list) >= max(per_month_cap or 0, 5_000):
                break
        df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=cols)
        if per_month_cap and len(df) > per_month_cap:
            df = df.sample(n=per_month_cap, random_state=RANDOM_SEED)

    if df.empty:
        return df

    # Nettoyage date + filtres
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])

    if exclude_on_events:
        df = df[df[PRODUCT_COL].astype(str).str.strip() != "Aucune_Proposition"]

    if df.empty:
        return df

    # Dédup stricte (client, date)
    df = (df.sort_values([CLIENT_ID_COL, TIME_COL])
            .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))
    return df


def build_events_stratified_across_months(
    limit_rows: int,
    months_back: int,
    per_month_cap: Optional[int],
    min_client_months: int,
    exclude_on_events: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Échantillonne plusieurs mois récents, applique une profondeur min par client,
    puis tronque à `limit_rows`. Si vide, relaxe progressivement pour garantir un retour non vide.
    """
    ds = dataiku.Dataset(DATASET_MAIN)
    months_all = _list_month_partitions(ds)
    if not months_all:
        raise RuntimeError("Impossible d'inférer les mois/partitions disponibles.")

    target_months = months_all[:max(1, months_back)]
    if not per_month_cap or per_month_cap <= 0:
        per_month_cap = max(100, limit_rows // max(1, len(target_months)))

    # Lecture stratifiée
    chunks = []
    total = 0
    for m in target_months:
        dfm = _read_month_sample(ds, m, per_month_cap=per_month_cap, exclude_on_events=exclude_on_events)
        if not dfm.empty:
            chunks.append(dfm)
            total += len(dfm)
        if total >= limit_rows * 2:
            break

    base_cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=base_cols)

    # --- Fallbacks garantissant le non-vide ---
    # F1: si vide, réessaie sans exclusion événements
    if df.empty and exclude_on_events:
        for m in target_months:
            dfm = _read_month_sample(ds, m, per_month_cap=per_month_cap, exclude_on_events=False)
            if not dfm.empty:
                chunks.append(dfm)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=base_cols)

    # F2: si encore vide, élargis à +months
    i = len(target_months)
    while df.empty and i < min(len(months_all), months_back * 3):
        m = months_all[i]
        dfm = _read_month_sample(ds, m, per_month_cap=per_month_cap, exclude_on_events=False)
        if not dfm.empty:
            chunks.append(dfm)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=base_cols)
        i += 1

    if df.empty:
        raise RuntimeError("df_events vide après tous les fallbacks (vérifie colonnes/données).")

    # Profondeur min par client
    tmp = df.copy()
    tmp["_m"] = tmp[TIME_COL].dt.to_period("M")
    keep_ids = tmp.groupby(CLIENT_ID_COL)["_m"].nunique()
    keep_ids = set(keep_ids[keep_ids >= min_client_months].index)
    df_kept = tmp[tmp[CLIENT_ID_COL].isin(keep_ids)].drop(columns=["_m"])

    # F3: si vide après filtre profondeur, relaxe à 1 mois min
    if df_kept.empty:
        df_kept = tmp.drop(columns=["_m"])

    # Équilibrage par mois + tronque à limit_rows
    df_kept["_month"] = df_kept[TIME_COL].dt.to_period("M").astype(str)
    n_target_months = df_kept["_month"].nunique()
    per_m = max(1, int(np.ceil(limit_rows / max(1, n_target_months))))
    df_kept = (df_kept.groupby("_month", group_keys=False).apply(lambda g: g.head(per_m)))
    df_kept = df_kept.drop(columns=["_month"]).head(limit_rows).reset_index(drop=True)

    n_clients = df_kept[CLIENT_ID_COL].nunique()
    n_months  = df_kept[TIME_COL].dt.to_period("M").nunique()

    meta = {
        "selected_months": target_months,
        "per_month_cap": per_month_cap,
        "exclude_on_events": exclude_on_events,
        "n_clients": int(n_clients),
        "n_months": int(n_months),
    }
    return df_kept, meta


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ===================== MAIN =========================
def main():
    np.random.seed(RANDOM_SEED)

    log.info("=== DRIVER START (in-memory, with non-empty guarantees) ===")
    log.info("== STEP 1: BUILD EVENTS (stratified) ==")

    # 1) Build events (avec garanties non-vides)
    try:
        df_events, meta = build_events_stratified_across_months(
            limit_rows=LIMIT_ROWS,
            months_back=MONTHS_BACK_TARGET,
            per_month_cap=PER_MONTH_CAP,
            min_client_months=MIN_CLIENT_MONTHS,
            exclude_on_events=EXCLUDE_ON_EVENTS
        )
    except Exception as e:
        log.error(f"BUILD EVENTS failed: {e}")
        raise

    log.info(f"Sample months (target): {meta['selected_months']}")
    log.info(f"df_events: rows={len(df_events):,} | clients={meta['n_clients']:,} | months={meta['n_months']}")
    try:
        vc = df_events[PRODUCT_COL].astype(str).value_counts().head(10)
        log.info(f"Top target values:\n{vc}")
    except Exception:
        pass

    # 2) Config (in-memory, no IO)
    log.info("== STEP 2: CONFIG (in-memory, no IO) ==")
    cfg = blank_config()

    # Données (on passe df_events en mémoire → pipeline ne relira pas Dataiku)
    cfg["data"]["events_df"]          = df_events
    cfg["data"]["events_dataset"]     = ""  # ignoré car events_df est fourni
    cfg["data"]["client_id_col"]      = CLIENT_ID_COL
    cfg["data"]["event_time_col"]     = TIME_COL
    cfg["data"]["product_col"]        = PRODUCT_COL
    cfg["data"]["event_extra_cols"]   = list(EXTRA_EVENT_COLS)

    # Profil OFF (pour RAM)
    cfg["data"]["dataset_name"]             = ""
    cfg["data"]["profile_categorical_cols"] = []
    cfg["data"]["profile_sequence_cols"]    = []
    cfg["data"]["profile_df"]               = None
    cfg["data"]["profile_join_key"]         = CLIENT_ID_COL

    # Séquence — on borne lookback par les mois réellement présents
    months_present = int(meta["n_months"])
    lookback = max(1, min(12, months_present))  # 12 si possible, sinon nb de mois dispo
    cfg["sequence"]["months_lookback"]          = lookback
    cfg["sequence"]["time_granularity"]         = "M"
    cfg["sequence"]["min_events_per_client"]    = 1
    cfg["sequence"]["target_horizon"]           = 1
    cfg["sequence"]["pad_value"]                = 0
    cfg["sequence"]["build_target_from_events"] = True

    # Features — on a déjà exclu côté événements → on ne double-pas ici
    cfg["features"]["exclude_target_values"] = []
    cfg["features"]["merge_rare_threshold"]  = MERGE_RARE_THRESHOLD
    cfg["features"]["other_class_name"]      = OTHER_CLASS_NAME

    # Modèle (petit)
    cfg["model"]["d_model"]             = D_MODEL
    cfg["model"]["n_heads"]             = N_HEADS
    cfg["model"]["n_layers"]            = N_LAYERS
    cfg["model"]["dropout"]             = 0.10
    cfg["model"]["max_sequence_length"] = lookback
    cfg["model"]["vocab_size"]          = 2000

    # Entraînement (CPU)
    cfg["training"]["batch_size"]      = BATCH_SIZE
    cfg["training"]["num_epochs"]      = EPOCHS
    cfg["training"]["learning_rate"]   = LEARNING_RATE
    cfg["training"]["weight_decay"]    = 1e-4
    cfg["training"]["val_split"]       = VAL_SPLIT
    cfg["training"]["class_weighting"] = True
    cfg["training"]["gradient_clip"]   = 1.0
    cfg["training"]["optimizer"]       = "adamw"

    # Sorties (on n’écrit pas les datasets ici ; juste un CSV local d’expériences)
    cfg["outputs"]["features_dataset"]        = None
    cfg["outputs"]["predictions_dataset"]     = None
    cfg["outputs"]["metrics_dataset"]         = None
    cfg["outputs"]["model_artifacts_dataset"] = None
    cfg["outputs"]["local_dir"]               = OUT_DIR

    cfg["runtime"]["verbose"]  = True
    cfg["runtime"]["progress"] = True
    cfg["runtime"]["seed"]     = RANDOM_SEED

    log.info(f"Archi: {cfg['model']['n_layers']}L-{cfg['model']['n_heads']}H-{cfg['model']['d_model']}D")
    log.info(f"Seq:   {cfg['sequence']['months_lookback']} mois | horizon={cfg['sequence']['target_horizon']}")
    log.info(f"Exclu (events level)? {EXCLUDE_ON_EVENTS}")
    log.info("OK config.")

    # 3) Train
    log.info("== STEP 3: TRAIN (CPU) ==")
    t0 = time.time()
    results = run_training(cfg)
    t_train = time.time() - t0
    log.info("Training finished.")
    log.info(f"Train time: {t_train:.1f}s")

    # 4) Metrics + experiment log CSV
    log.info("== STEP 4: METRICS ==")
    m = results.get("metrics", {})
    mi = results.get("model_info", {})
    di = results.get("data_info", {})

    acc = float(m.get("accuracy", 0.0))
    prec = float(m.get("precision", 0.0))
    rec = float(m.get("recall", 0.0))
    f1  = float(m.get("f1", 0.0))

    log.info(f"accuracy  : {acc:.4f}")
    log.info(f"precision : {prec:.4f}")
    log.info(f"recall    : {rec:.4f}")
    log.info(f"f1        : {f1:.4f}")
    log.info(f"Model: {mi.get('architecture','N/A')} | params≈ {mi.get('total_params','NA'):,}")
    log.info(f"Data : clients={di.get('n_clients','NA')} | seq_len={di.get('seq_len','NA')} | classes={di.get('n_classes','NA')}")

    # CSV d’expériences
    ensure_dir(OUT_DIR)
    row = {
        "timestamp": datetime.now().isoformat(),
        "rows": len(df_events),
        "clients": int(meta["n_clients"]),
        "months": int(meta["n_months"]),
        "selected_months_head": ";".join(map(str, meta["selected_months"][:3])),
        "lookback": lookback,
        "exclude_on_events": int(EXCLUDE_ON_EVENTS),
        "merge_rare_threshold": MERGE_RARE_THRESHOLD,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "val_split": VAL_SPLIT,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_classes": int(di.get("n_classes", -1)),
        "seq_len": int(di.get("seq_len", -1)),
        "train_time_s": float(t_train),
    }
    df_log = pd.DataFrame([row])
    if os.path.exists(EXP_LOG_CSV):
        df_log.to_csv(EXP_LOG_CSV, index=False, mode="a", header=False)
    else:
        df_log.to_csv(EXP_LOG_CSV, index=False)
    log.info(f"Experiment row appended to {EXP_LOG_CSV}")

    log.info("=== DRIVER DONE ===")


if __name__ == "__main__":
    main()



