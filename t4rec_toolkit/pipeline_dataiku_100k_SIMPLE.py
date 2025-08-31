#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD EVENTS (in-memory, robust, with LIMIT and full-count)
- Zéro appel à get_info()/get_schema()
- Détecte les colonnes via get_dataframe(limit=0)
- Lecture par chunks sans parse_dates automatique (évite bugs d'index)
- Normalise DATMAJ en tz-naive, filtre dernière N mois, déduplique (client, date)
- Agrège jusqu'à LIMIT_MAX lignes, puis continue en "count only" pour donner le total sans limite
"""

import dataiku
import pandas as pd
from datetime import datetime

# ------------ CONFIG ------------
DATASET_MAIN     = "BASE_SCORE_COMPLETE_prepared"
CLIENT_ID_COL    = "NUMTECPRS"                    # id client (remplacé demandé)
TIME_COL         = "DATMAJ"                       # datetime
PRODUCT_COL      = "SOUSCRIPTION_PRODUIT_1M"      # produit / item
EXTRA_EVENT_COLS = []                             
KEEP_MONTHS      = 24                             # garder les N derniers mois
CHUNKSIZE        = 200_000                        # chunk size
LIMIT_MAX        = 3_000_000                      # limite d'agrégation en mémoire
PROGRESS_EVERY   = 200_000                        # fréquence d'affichage progression
# --------------------------------

print("== BUILD EVENTS (memory) ==", datetime.now())

src = dataiku.Dataset(DATASET_MAIN)

# Récupération "safe" des colonnes disponibles
try:
    schema_cols = list(src.get_dataframe(limit=0).columns)
except Exception:
    # vieux DSS : lire la conf pour retrouver les colonnes déclarées
    cfg = src.get_config()
    schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
missing = [c for c in cols_needed if c not in schema_cols]
if missing:
    raise ValueError(f"Colonnes absentes dans {DATASET_MAIN}: {missing}")

keep_set = set(cols_needed)

# Compteurs
rows_kept_total_no_limit = 0   # après filtres & dédupes, si on ne limitait pas
rows_kept_aggregated     = 0   # réellement agrégées dans df_events (≤ LIMIT_MAX)
rows_scanned_raw         = 0   # lignes brutes lues (avant filtres)

chunks = []
print_every_next = PROGRESS_EVERY

for chunk in src.iter_dataframes(
    chunksize=CHUNKSIZE,
    parse_dates=False,          # pas de parse auto => on contrôle
    infer_with_pandas=True
):
    rows_scanned_raw += len(chunk)

    # 1) Trim aux colonnes utiles
    inter = [c for c in chunk.columns if c in keep_set]
    chunk = chunk[inter]

    # 2) Drop NA id/date
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])
    if chunk.empty:
        # progress
        if rows_scanned_raw >= print_every_next:
            print(f" scanned={rows_scanned_raw:,} | kept(no-limit)={rows_kept_total_no_limit:,} | aggregated={rows_kept_aggregated:,}")
            print_every_next += PROGRESS_EVERY
        continue

    # 3) Parse DATMAJ → tz-aware → strip tz → tz-naive
    dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)
    chunk[TIME_COL] = dt
    chunk = chunk.dropna(subset=[TIME_COL])
    if chunk.empty:
        if rows_scanned_raw >= print_every_next:
            print(f" scanned={rows_scanned_raw:,} | kept(no-limit)={rows_kept_total_no_limit:,} | aggregated={rows_kept_aggregated:,}")
            print_every_next += PROGRESS_EVERY
        continue

    # 4) Filtre temporel (dernier N mois)
    if KEEP_MONTHS is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS)
        chunk = chunk[chunk[TIME_COL] >= cutoff]
        if chunk.empty:
            if rows_scanned_raw >= print_every_next:
                print(f" scanned={rows_scanned_raw:,} | kept(no-limit)={rows_kept_total_no_limit:,} | aggregated={rows_kept_aggregated:,}")
                print_every_next += PROGRESS_EVERY
            continue

    # 5) Dédupe (client, date) en gardant la dernière
    chunk = (
        chunk.sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
    )

    # Compte "no-limit" d'abord
    rows_kept_total_no_limit += len(chunk)

    # 6) Agrégation sous LIMIT_MAX
    remaining = max(0, LIMIT_MAX - rows_kept_aggregated)
    if remaining > 0:
        if len(chunk) <= remaining:
            chunks.append(chunk)
            rows_kept_aggregated += len(chunk)
        else:
            # on prend juste la part qui rentre dans la limite
            chunks.append(chunk.iloc[:remaining].copy())
            rows_kept_aggregated += remaining
            # le reste du chunk continue d'être compté via rows_kept_total_no_limit (déjà fait)

    # Progression périodique
    if rows_scanned_raw >= print_every_next:
        pct = (rows_kept_aggregated / LIMIT_MAX * 100.0) if LIMIT_MAX > 0 else 0.0
        print(f" scanned={rows_scanned_raw:,} | kept(no-limit)={rows_kept_total_no_limit:,} | aggregated={rows_kept_aggregated:,} ({pct:.1f}% of LIMIT)")
        print_every_next += PROGRESS_EVERY

# Concat final
import pandas as pd as _pd  # avoid shadowing if any
df_events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_needed)

print("== DONE ==")
print(f"Raw scanned rows      : {rows_scanned_raw:,}")
print(f"Kept (no limit) rows  : {rows_kept_total_no_limit:,}   # après filtres & dédupes")
print(f"Aggregated (LIMIT)    : {rows_kept_aggregated:,} / {LIMIT_MAX:,}")
print("df_events shape       :", df_events.shape)
print(df_events.head(10))


