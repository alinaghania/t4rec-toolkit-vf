# === NOTEBOOK CELL : BUILD EVENTS DATASET (STREAMED, FIXED) ===
import pandas as pd
import numpy as np
from datetime import datetime
import dataiku

# ---- Paramètres
DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"

CLIENT_ID_COL  = "CLIENT_ID"
TIME_COL       = "DATMAJ"                    # Datetime-like (ex: 2024-07-31)
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []                        # ex: ["CANAL", "FAMILLE"]

TARGET_EXCLUDE = {"aucune_souscription"}     # réduit la taille
CHUNK_ROWS     = 200_000                     # buffer RAM
ASSUME_PARTITIONED = True                    # si le dataset cible est partitionné (yyyy-MM)

print("== BUILD EVENTS (stream) ==")
print(datetime.now())

src = dataiku.Dataset(DATASET_MAIN)
dst = dataiku.Dataset(DATASET_EVENTS)

# --- Vérif colonnes (fix API) ---
schema_cols = [c["name"] for c in src.read_schema()]  # <== CORRIGÉ
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + [c for c in EXTRA_EVENT_COLS]
missing = [c for c in cols_needed if c not in schema_cols]
if missing:
    raise ValueError(f"Colonnes absentes dans {DATASET_MAIN}: {missing}")

# --- Writer helper (gère partitionné / non partitionné) ---
def write_chunk(df: pd.DataFrame):
    """Ecrit un chunk dans DATASET_EVENTS. On suppose partition mensuelle 'yyyy-MM' si possible."""
    if df.empty:
        return
    df = df.copy()
    # bucket mensuel
    df["_bucket"] = pd.to_datetime(df[TIME_COL], errors="coerce").dt.to_period("M").dt.to_timestamp()
    # dédup par (client, mois) en gardant la ligne la plus récente
    df = (df.sort_values([CLIENT_ID_COL, TIME_COL])
            .drop_duplicates(subset=[CLIENT_ID_COL, "_bucket"], keep="last"))
    # partition string
    df["_part"] = df["_bucket"].dt.strftime("%Y-%m")

    # colonnes finales à écrire
    to_write = df.drop(columns=["_bucket"]).copy()

    if ASSUME_PARTITIONED:
        # on tente le mode partitionné; si ça explose, on fallback non-partitionné
        try:
            for part, sub in to_write.groupby("_part"):
                with dst.get_writer(partition=str(part)) as w:
                    for row in sub.drop(columns=["_part"]).to_dict(orient="records"):
                        w.write_row_dict(row)
            return
        except Exception as e:
            print(f"⚠️ Partition write failed ({e}); fallback en non partitionné.")
            # on continue en non partitionné

    # non partitionné
    with dst.get_writer() as w:
        for row in to_write.drop(columns=["_part"]).to_dict(orient="records"):
            w.write_row_dict(row)

# --- Boucle stream ---
buf = []
rows_in_total = 0
rows_out_total = 0
select_cols = cols_needed  # lecture restreinte

for row in src.iter_rows(columns=select_cols):
    buf.append(row)
    if len(buf) >= CHUNK_ROWS:
        df = pd.DataFrame(buf)
        buf.clear()

        # nettoyage minimal
        df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL])

        # filtre label (réduit volume)
        df = df[~df[PRODUCT_COL].astype(str).str.lower().isin(TARGET_EXCLUDE)]

        write_chunk(df)
        rows_in_total += len(df)
        rows_out_total += len(df)
        print(f"  wrote chunk — rows_in_total={rows_in_total:,}")

# flush final
if buf:
    df = pd.DataFrame(buf)
    buf.clear()
    df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df[~df[PRODUCT_COL].astype(str).str.lower().isin(TARGET_EXCLUDE)]
    write_chunk(df)
    rows_in_total += len(df)
    rows_out_total += len(df)

print("== DONE ==")
print(f"Rows written (before de-dup per month): {rows_out_total:,}")

