"""
BUILD EVENTS (memory)
----------------------
- Charge un dataset Dataiku brut (6M lignes / 560 colonnes).
- Construit une vue "événements" avec colonnes : client_id, time, produit (+ extras).
- Nettoie les doublons et garde la dernière interaction par (client, mois).
- Stocke le résultat dans un DataFrame temporaire `events_df` utilisable pour le modèle.
"""

import dataiku
import pandas as pd
from datetime import datetime

# ================== PARAMÈTRES ==================
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"  # dataset brut
CLIENT_ID_COL = "NUMTECPRS"                    # identifiant client
TIME_COL = "DATMAJ"                            # colonne temporelle
PRODUCT_COL = "SOUSCRIPTION_PRODUIT_1M"        # colonne produit cible
EXTRA_EVENT_COLS = []                          # éventuelles colonnes supplémentaires
KEEP_MONTHS = 24                               # fenêtre temporelle max (24 derniers mois)
CHUNKSIZE = 100_000                            # taille des batchs (ajuste si OOM)

print("== BUILD EVENTS (memory) ==", datetime.now())

# ================== CHARGEMENT STREAMING ==================
src = dataiku.Dataset(DATASET_MAIN)

cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + [
    c for c in EXTRA_EVENT_COLS if c in [col["name"] for col in src.read_schema()]
]

events_chunks = []
rows_total = 0

for chunk in src.iter_dataframes(chunksize=CHUNKSIZE, columns=cols_needed):
    # Nettoyage de base
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    # Uniformise la colonne temps → UTC naive
    chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True).dt.tz_localize(None)
    chunk = chunk.dropna(subset=[TIME_COL])

    # Restriction aux derniers KEEP_MONTHS mois
    if KEEP_MONTHS is not None:
        cutoff = (pd.Timestamp.utcnow().normalize() - pd.DateOffset(months=KEEP_MONTHS))
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    events_chunks.append(chunk)
    rows_total += len(chunk)
    print(f"  → Lu {rows_total:,} lignes cumulées")

# Fusion de tous les morceaux
events_df = pd.concat(events_chunks, ignore_index=True)
print(f"[OK] Events fusionnés : {events_df.shape}")

# ================== DÉDUPLICATION ==================
# Garde la dernière interaction par (client, mois)
events_df = (events_df
             .sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))

print(f"[OK] Events après déduplication : {events_df.shape}")

# ================== RÉSULTAT ==================
print(events_df.head(10))
