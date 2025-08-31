# === NOTEBOOK CELL : BUILD EVENTS DATASET (STREAMED) ===
import pandas as pd
import numpy as np
from datetime import datetime
import dataiku

# ---- Paramètres
DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"

CLIENT_ID_COL  = "CLIENT_ID"
TIME_COL       = "DATMAJ"                   # Datetime-like
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []                       # ex: ["CANAL", "FAMILLE"]

TARGET_EXCLUDE = {"aucune_souscription"}    # on peut filtrer pour réduire la taille
CHUNK_ROWS     = 200_000                    # taille tampon RAM
WRITE_PARTITIONED = True                    # si le dataset cible est partitionné par mois (yyyy-MM)

print("== BUILD EVENTS (stream) ==")
print(datetime.now())

src = dataiku.Dataset(DATASET_MAIN)
dst = dataiku.Dataset(DATASET_EVENTS)

cols = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + [c for c in EXTRA_EVENT_COLS]
missing = [c for c in cols if c not in src.read_schema_columns()]
if missing:
    raise ValueError(f"Colonnes absentes dans {DATASET_MAIN}: {missing}")

# Petit helper : écrit un DataFrame vers le dataset cible, en partition si possible
def write_chunk(df: pd.DataFrame):
    if df.empty:
        return
    # Convertit DATMAJ en début de mois + string partition
    df["_bucket"] = pd.to_datetime(df[TIME_COL], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if WRITE_PARTITIONED:
        df["_part"] = df["_bucket"].dt.strftime("%Y-%m")
        for part, sub in df.groupby("_part"):
            with dst.get_writer(partition=str(part)) as w:
                for row in sub.drop(columns=["_bucket", "_part"]).to_dict(orient="records"):
                    w.write_row_dict(row)
    else:
        # Append non partitionné
        with dst.get_writer() as w:
            for row in df.drop(columns=["_bucket"]).to_dict(orient="records"):
                w.write_row_dict(row)

# Tampon
buf = []
buf_count_in = 0
buf_count_out = 0

# Itération “stream” (pas de gros DataFrame en mémoire)
for row in src.iter_rows(columns=cols):  # génère dict par ligne
    buf.append(row)
    if len(buf) >= CHUNK_ROWS:
        df = pd.DataFrame(buf)
        buf.clear()

        # Nettoyage minimal
        df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL])

        # (optionnel) filtrer les non-souscriptions
        df = df[~df[PRODUCT_COL].astype(str).str.lower().isin(TARGET_EXCLUDE)]

        # Garder 1 ligne par (client, mois) → la plus récente
        df = df.sort_values([CLIENT_ID_COL, TIME_COL]).drop_duplicates(
            subset=[CLIENT_ID_COL, TIME_COL], keep="last"
        )

        # Écriture
        write_chunk(df)

        buf_count_in += len(df)
        buf_count_out += len(df)
        print(f"  wrote chunk — rows_in={buf_count_in:,}")

# Dernier flush
if buf:
    df = pd.DataFrame(buf)
    buf.clear()

    df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df[~df[PRODUCT_COL].astype(str).str.lower().isin(TARGET_EXCLUDE)]
    df = df.sort_values([CLIENT_ID_COL, TIME_COL]).drop_duplicates(
        subset=[CLIENT_ID_COL, TIME_COL], keep="last"
    )
    write_chunk(df)
    buf_count_in += len(df)
    buf_count_out += len(df)

print("== DONE ==")
print(f"Rows written: {buf_count_out:,}")
