# == BUILD EVENTS (stream) — robuste, sans get_schema ==
from datetime import datetime
import pandas as pd
import dataiku

print("== BUILD EVENTS (stream) ==", datetime.now())

# --- Paramètres requis ---
SRC_NAME   = DATASET_MAIN                 # ex: "BASE_SCORE_COMPLETE_prepared"
DST_NAME   = DATASET_EVENTS               # ex: "T4REC_EVENTS_FROM_MAIN"
CID        = CLIENT_ID_COL                # ex: "CLIENT_ID"
TCOL       = TIME_COL                     # ex: "DATMAJ"
PCOL       = PRODUCT_COL                  # ex: "SOUSCRIPTION_PRODUIT_1M"
EXTRA_COLS = list(EXTRA_EVENT_COLS)       # ex: []

# --- Ouverture datasets ---
src = dataiku.Dataset(SRC_NAME)
dst = dataiku.Dataset(DST_NAME)

# --- Récupère la liste des colonnes de façon 100% compatible versions ---
# 1) essai via get_config (toujours dispo)
try:
    cfg = src.get_config()
    cols_in_schema = [c["name"] for c in cfg.get("schema", {}).get("columns", [])]
except Exception:
    cols_in_schema = []
# 2) fallback ultra simple : lit 1 ligne pour obtenir les noms (ne consomme pas de RAM)
if not cols_in_schema:
    cols_in_schema = list(src.get_dataframe(limit=1).columns)

# --- Vérifications colonnes nécessaires ---
needed = [CID, TCOL, PCOL] + [c for c in EXTRA_COLS]
missing = [c for c in needed if c not in cols_in_schema]
if missing:
    raise ValueError(f"Colonnes absentes dans {SRC_NAME}: {missing}")

# --- Fonction utilitaire : normalisation/clean d'un chunk ---
def prepare_chunk(df):
    # Garde seulement les colonnes utiles
    df = df[needed].copy()

    # Nettoyage minimal
    df = df.dropna(subset=[CID, TCOL])
    df[TCOL] = pd.to_datetime(df[TCOL], errors="coerce")
    df = df.dropna(subset=[TCOL])

    # IMPORTANT: on ne déduplique que **dans le chunk** pour réduire le volume ;
    # d’éventuels doublons (CID, TCOL) à cheval sur 2 chunks seront gérés
    # plus loin par le SequenceBuilder (qui prend le dernier évènement du bucket).
    df = df.sort_values([CID, TCOL]).drop_duplicates(subset=[CID, TCOL], keep="last")
    return df

# --- Écriture streaming : premier chunk = write_with_schema, suivants = append ---
CHUNK = 100_000  # ajuste si besoin (50k~200k selon RAM/format)
first = True
n_rows_out = 0
n_chunks = 0

# iter_dataframes est la voie normale pour streamer sans OOM
if hasattr(src, "iter_dataframes"):
    for raw in src.iter_dataframes(chunksize=CHUNK):
        n_chunks += 1
        df_chunk = prepare_chunk(raw)
        if df_chunk.empty:
            continue
        if first:
            # crée/écrase le dataset de sortie avec le bon schéma automatiquement
            dst.write_with_schema(df_chunk)
            first = False
        else:
            # append performant (si write_dataframe indisponible on bascule en lignes)
            w = dst.get_writer()
            try:
                w.write_dataframe(df_chunk)
            except Exception:
                for rec in df_chunk.to_dict(orient="records"):
                    w.write_row_dict(rec)
            finally:
                w.close()
        n_rows_out += len(df_chunk)
        print(f"  wrote chunk {n_chunks:,} → {len(df_chunk):,} rows (total {n_rows_out:,})")
else:
    # Fallback ultime (versions très anciennes): lecture complète (risque d'OOM).
    # Utilise-le seulement sur un sous-ensemble (ex: limit=1_000_000) si RAM limitée.
    print("⚠️ iter_dataframes() indisponible: fallback lecture complète (attention RAM).")
    raw = src.get_dataframe()  # <-- à éviter si > RAM
    df_chunk = prepare_chunk(raw)
    dst.write_with_schema(df_chunk)
    n_chunks = 1
    n_rows_out = len(df_chunk)

print(f"✅ EVENTS BUILT → {DST_NAME}: {n_rows_out:,} rows in {n_chunks} chunk(s)")



