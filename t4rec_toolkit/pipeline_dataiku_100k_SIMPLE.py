# == BUILD EVENTS (stream) ==
from datetime import datetime
import pandas as pd
import dataiku

DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"
CLIENT_ID_COL  = "CLIENT_ID"
TIME_COL       = "DATMAJ"
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []  # ex: ["CANAL", "FAMILLE_PRODUIT"]

print("== BUILD EVENTS (stream) ==")
print(datetime.now())

src = dataiku.Dataset(DATASET_MAIN)
dst = dataiku.Dataset(DATASET_EVENTS)

# 1) Colonnes du schéma source
src_schema = src.get_schema()
src_cols = [c["name"] for c in src_schema["columns"]]
needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
missing = [c for c in needed if c not in src_cols]
if missing:
    raise ValueError(f"Colonnes absentes dans {DATASET_MAIN}: {missing}")

# 2) S’assurer que le schéma cible existe (sinon créer un schéma minimal en écrivant 0 ligne)
try:
    _ = dst.get_schema()  # essaie de lire le schéma
except Exception:
    # crée un schéma minimal en écrivant un DF vide
    empty_df = pd.DataFrame(columns=[CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS))
    dst.write_with_schema(empty_df)

# 3) On va bufferiser par (client, mois) pour garder uniquement la dernière occurrence
#    -> dictionnaire clé=(client_id, month_start) -> dict ligne compacte
from dateutil.relativedelta import relativedelta

buffer_last = {}  # {(cid, month_ts): row_dict_compact}

# 4) Lecture en flux
col_index = {name: idx for idx, name in enumerate(src_cols)}  # pour indexer rapidement le tuple
for row_tuple in src.iter_rows():                              # tuples dans l'ordre des colonnes du schéma
    # -- extraire uniquement les colonnes utiles
    try:
        cid = row_tuple[col_index[CLIENT_ID_COL]]
        ts  = row_tuple[col_index[TIME_COL]]
        prod = row_tuple[col_index[PRODUCT_COL]]
    except Exception:
        continue  # ligne anormale → skip

    if cid is None or ts is None:
        continue

    # -- parse date
    try:
        ts = pd.to_datetime(ts, errors="coerce")
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        continue

    # -- bucket mensuel (début de mois)
    month_ts = ts.to_period("M").to_timestamp()

    # -- récupérer extras (si demandés)
    row_out = {
        CLIENT_ID_COL: cid,
        TIME_COL: month_ts,
        PRODUCT_COL: prod
    }
    for c in EXTRA_EVENT_COLS:
        # si la colonne est présente, on la copie telle quelle
        if c in col_index:
            row_out[c] = row_tuple[col_index[c]]

    # -- on garde la plus récente pour (client, mois) : on remplace systématiquement
    buffer_last[(cid, month_ts)] = row_out

# 5) Écriture en flux des lignes agrégées
with dst.get_writer() as w:
    for _, row_out in buffer_last.items():
        w.write_row_dict(row_out)

print(f"✅ Events écrits dans {DATASET_EVENTS} : {len(buffer_last)} lignes")


