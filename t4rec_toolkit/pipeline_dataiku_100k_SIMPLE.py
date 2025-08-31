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

