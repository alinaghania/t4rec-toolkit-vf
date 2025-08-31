# == BUILD EVENTS (pandas) ==
from datetime import datetime
import pandas as pd
import dataiku

# --- Paramètres (reprends tes constantes)
DATASET_MAIN   = "BASE_SCORE_COMPLETE_prepared"
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"
CLIENT_ID_COL  = "CLIENT_ID"
TIME_COL       = "DATMAJ"
PRODUCT_COL    = "SOUSCRIPTION_PRODUIT_1M"
EXTRA_EVENT_COLS = []  # ex: ["CANAL", "FAMILLE_PRODUIT"]

print("== BUILD EVENTS (pandas) ==")
print(datetime.now())

# 1) Source + vérification des colonnes
src = dataiku.Dataset(DATASET_MAIN)
schema_cols = [c["name"] for c in src.get_schema()["columns"]]  # noms des colonnes du schéma
needed = [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + list(EXTRA_EVENT_COLS)
missing = [c for c in needed if c not in schema_cols]
if missing:
    raise ValueError(f"Colonnes absentes dans {DATASET_MAIN}: {missing}")

# 2) Charger seulement les colonnes utiles (RAM friendly vs full table)
df = src.get_dataframe(columns=needed)

# 3) Nettoyage de base
df = df.dropna(subset=[CLIENT_ID_COL, TIME_COL])                       # garde lignes avec id + date
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")           # parse datetime
df = df.dropna(subset=[TIME_COL])                                      # retire dates invalides

# 4) Bucket mensuel (début de mois) pour la dimension temporelle
df["_month"] = df[TIME_COL].dt.to_period("M").dt.to_timestamp()        # 1er jour du mois

# 5) 1 ligne par (client, mois) : garder la plus récente si plusieurs
df = (df.sort_values([CLIENT_ID_COL, TIME_COL])                        # ordre chronologique
        .drop_duplicates(subset=[CLIENT_ID_COL, "_month"], keep="last"))

# 6) Renommer proprement pour la table événements
events = df[[CLIENT_ID_COL, "_month", PRODUCT_COL] + [c for c in EXTRA_EVENT_COLS if c in df.columns]].copy()
events = events.rename(columns={"_month": TIME_COL})                    # la colonne de temps redevient TIME_COL

# 7) Écriture/sauvegarde
dst = dataiku.Dataset(DATASET_EVENTS)
dst.write_with_schema(events)                                           # crée/écrase avec le schéma du DF

print(f"✅ Events écrits dans {DATASET_EVENTS} : {events.shape}")



