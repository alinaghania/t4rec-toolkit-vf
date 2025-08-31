#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD EVENTS (in-memory)
========================
- Lit le dataset principal depuis Dataiku
- Garde uniquement (NUMTECPRS, DATMAJ, SOUSCRIPTION_PRODUIT_1M)
- Nettoie, filtre par date, garde 1 ligne par (client, mois)
- Concatène tous les chunks → df_events utilisable directement
"""

import dataiku
import pandas as pd
from datetime import datetime

# ---------------- CONFIG ----------------
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"   # ton dataset principal (6M lignes)
CLIENT_ID_COL = "NUMTECPRS"                     # identifiant client
TIME_COL = "DATMAJ"                             # colonne temporelle
PRODUCT_COL = "SOUSCRIPTION_PRODUIT_1M"         # produit cible
EXTRA_EVENT_COLS = []                           # ex: ["CANAL", "FAMILLE"]

KEEP_MONTHS = 24                                # n derniers mois à garder
CHUNKSIZE = 200_000                             # taille de chunk (mémoire)
# -----------------------------------------

print("== BUILD EVENTS (memory) ==", datetime.now())

# 1) Colonnes nécessaires
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_EVENT_COLS

# 2) Init source Dataiku
src = dataiku.Dataset(DATASET_MAIN)

# 3) Lecture par chunks
chunks = []
rows_total = 0

for chunk in src.iter_dataframes(chunksize=CHUNKSIZE, columns=cols_needed):
    # -- Nettoyage de base
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True).dt.tz_localize(None)
    chunk = chunk.dropna(subset=[TIME_COL])

    # -- Filtre par date (KEEP_MONTHS)
    if KEEP_MONTHS is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS)
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    # -- Garde la dernière ligne par (client, date)
    chunk = (chunk
             .sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))

    chunks.append(chunk)
    rows_total += len(chunk)
    print(f"   + {len(chunk):,} lignes (cumul={rows_total:,})")

# 4) Concaténation finale
if len(chunks) > 0:
    df_events = pd.concat(chunks, ignore_index=True)
else:
    df_events = pd.DataFrame(columns=cols_needed)

print("== DONE ==")
print(f"Final events shape : {df_events.shape}")
print(df_events.head(10))

