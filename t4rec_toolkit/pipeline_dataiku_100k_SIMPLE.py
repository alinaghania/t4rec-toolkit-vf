#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD EVENTS (in-memory, robust to DSS/Pandas parser quirks)
- Reads the main dataset in chunks without 'columns=' to avoid parse_dates index bugs
- Immediately trims to needed cols in-memory
- Parses DATMAJ to tz-naive, filters last N months, dedups (client, date)
- Produces df_events for the model
"""

import dataiku
import pandas as pd
from datetime import datetime

# ---------------- CONFIG ----------------
DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"   # main dataset
CLIENT_ID_COL  = "NUMTECPRS"                      # client id
TIME_COL       = "DATMAJ"                         # datetime column
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"        # product / item
EXTRA_EVENT_COLS = []                              # e.g., ["CANAL", "FAMILLE"]
KEEP_MONTHS    = 24                                # last N months
CHUNKSIZE      = 200_000                           # tune if memory tight
# -----------------------------------------

print("== BUILD EVENTS (memory) ==", datetime.now())

# Instantiate dataset
src = dataiku.Dataset(DATASET_MAIN)

# Discover available columns safely (empty DF with schema)
try:
    schema_cols = list(src.get_dataframe(limit=0).columns)
except Exception:
    # very old DSS fallback: use get_config
    cfg = src.get_config()
    schema_cols = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]

# Needed & available
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_EVENT_COLS
missing = [c for c in cols_needed if c not in schema_cols]
if missing:
    raise ValueError(f"Missing columns in {DATASET_MAIN}: {missing}")

keep_set = set(cols_needed)

# Chunked read WITHOUT 'columns=' and WITHOUT automatic date parsing
chunks = []
rows_total = 0

for chunk in src.iter_dataframes(
    chunksize=CHUNKSIZE,
    parse_dates=False,          # avoid DSS↔pandas index mapping issues
    infer_with_pandas=True
):
    # Trim to needed columns ASAP (ignoring any unexpected extras)
    inter = [c for c in chunk.columns if c in keep_set]
    chunk = chunk[inter]

    # Basic cleaning
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])

    # Robust datetime parse → tz-aware → drop tz (naive)
    # (handles strings, mixed, already-datetime)
    dt = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None) if hasattr(dt, "dt") else dt
    chunk[TIME_COL] = dt
    chunk = chunk.dropna(subset=[TIME_COL])

    # Filter last N months (cutoff tz-naive)
    if KEEP_MONTHS is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=KEEP_MONTHS)
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    # Dedup: keep last row per (client, date)
    chunk = (
        chunk.sort_values([CLIENT_ID_COL, TIME_COL])
             .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
    )

    if not chunk.empty:
        chunks.append(chunk)
        rows_total += len(chunk)
        print(f"   + {len(chunk):,} rows (cumulative={rows_total:,})")

# Final concat
df_events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_needed)

print("== DONE ==")
print(f"Final events shape : {df_events.shape}")
print(df_events.head(10))

