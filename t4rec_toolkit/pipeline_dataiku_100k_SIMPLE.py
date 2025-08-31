# == BUILD EVENTS (memory) ==
from datetime import datetime
import pandas as pd
import dataiku

print("== BUILD EVENTS (memory) ==", datetime.now())

# --- Paramètres ---
SRC_NAME   = "BASE_SCORE_COMPLETE_prepared"   # dataset source principal
CLIENT_ID_COL = "NUMTECPRS"                   # identifiant client
TIME_COL     = "DATMAJ"                       # colonne temporelle
PRODUCT_COL  = "SOUSCRIPTION_PRODUIT_1M"      # produit du mois
EXTRA_COLS   = []                             # ex: ["CANAL", "MONTANT"]

KEEP_MONTHS  = 24       # fenêtre temporelle
CHUNKSIZE    = 200_000  # taille de chunk (RAM friendly)
MAX_ROWS_HINT= 1_000_000 # limite soft pour debug (None = tout)

# --- Ouverture dataset Dataiku ---
src = dataiku.Dataset(SRC_NAME)

# --- Colonnes nécessaires ---
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_COLS

# --- Build DataFrame en mémoire (streaming chunks) ---
events_parts = []
rows_total = 0

for chunk in src.iter_dataframes(
    chunksize=CHUNKSIZE,
    columns=cols_needed,
    infer_with_pandas=True,
    parse_dates=False   # ⚠️ évite bug IndexError
):
    # Nettoyage minimal
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])

    # Cast manuel en datetime
    chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
    chunk = chunk.dropna(subset=[TIME_COL])

    # Filtre fenêtre temporelle (24 mois par défaut)
    if KEEP_MONTHS is not None:
        cutoff = (pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS))
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    # Garde la dernière ligne par (client, date)
    chunk = (chunk
             .sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))

    events_parts.append(chunk)
    rows_total += len(chunk)

    print(f"  Chunk {len(events_parts):,} → {len(chunk):,} rows (total {rows_total:,})")

    if MAX_ROWS_HINT is not None and rows_total >= MAX_ROWS_HINT:
        break

# --- Concat final ---
events_df = pd.concat(events_parts, ignore_index=True)
print(f"✅ EVENTS BUILT → {events_df.shape} (sur {rows_total:,} lignes traitées)")

