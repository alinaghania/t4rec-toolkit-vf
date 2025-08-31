#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD EVENTS (memory) — robust to Dataiku parse_dates/index issues
- Streams all columns (no columns=) to avoid pandas index mismatch
- Disables DSS auto date parsing (parse_dates=False)
- Manually parses TIME_COL to UTC-naive
"""

import dataiku
import pandas as pd
from datetime import datetime

# ========= Params =========
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"
CLIENT_ID_COL = "NUMTECPRS"
TIME_COL = "DATMAJ"
PRODUCT_COL = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []
KEEP_MONTHS = 24
CHUNKSIZE = 100_000  # lower if you still hit memory limits

print("== BUILD EVENTS (memory) ==", datetime.now())

src = dataiku.Dataset(DATASET_MAIN)

# We'll stream ALL columns to avoid pandas/date index mismatch
# and turn off DSS date parsing, then select what we need.
cols_needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + EXTRA_EVENT_COLS

events_chunks = []
rows_total = 0
chunk_id = 0

for chunk in src.iter_dataframes(
        chunksize=CHUNKSIZE,
        parse_dates=False,          # <- CRUCIAL: avoid DSS/pandas date-index parsing
        infer_with_pandas=True      # use pandas dtypes
    ):
    chunk_id += 1

    # Keep only the required columns that actually exist
    present = [c for c in cols_needed if c in chunk.columns]
    missing_now = [c for c in cols_needed if c not in chunk.columns]
    if missing_now:
        # If a required column is missing from this chunk, skip it (rare, but safe)
        print(f"  [chunk {chunk_id}] Missing cols {missing_now} -> skipping rows for this chunk")
        continue

    chunk = chunk[present]

    # Basic cleaning
    chunk = chunk.dropna(subset=[CLIENT_ID_COL, TIME_COL])

    # Manual, robust time parsing → UTC-naive
    chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True).dt.tz_localize(None)
    chunk = chunk.dropna(subset=[TIME_COL])

    # Filter to last KEEP_MONTHS months
    if KEEP_MONTHS is not None:
        cutoff = (pd.Timestamp.utcnow().normalize() - pd.DateOffset(months=KEEP_MONTHS))
        chunk = chunk[chunk[TIME_COL] >= cutoff]

    # Append if non-empty
    if not chunk.empty:
        events_chunks.append(chunk)
        rows_total += len(chunk)
        print(f"  → {rows_total:,} rows cumulated (after cleaning)")

# Concatenate all chunks
if events_chunks:
    events_df = pd.concat(events_chunks, ignore_index=True)
else:
    events_df = pd.DataFrame(columns=cols_needed)

print(f"[OK] Events merged: {events_df.shape}")

# Deduplicate: keep last record per (client, exact date)
if not events_df.empty:
    events_df = (
        events_df
        .sort_values([CLIENT_ID_COL, TIME_COL])
        .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last")
        .reset_index(drop=True)
    )

print(f"[OK] After dedup: {events_df.shape}")
print(events_df.head(10))
