#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# NOTEBOOK T4REC — PIPELINE TEMPOREL (FS => CONFIG => TRAIN => METRICS)
# - Utilise BASE_SCORE_COMPLETE_prepared comme source unique
# - Construit automatiquement une vue "événements" (client, mois, produit)
# - Entraîne le modèle hybride (T4Rec embeddings + Transformer PyTorch)
# - Sauvegarde Features/Prédictions/Métriques dans Dataiku (si configurés)
# ===============================================================

# -------- Imports standards --------
import os  # gestion de chemins (facultatif)
import json  # pour logs/mappings (facultatif)
import pandas as pd  # manipulation de données
import numpy as np  # calculs numériques

# -------- Imports Dataiku --------
import dataiku  # pour lire/écrire les datasets dans Dataiku

# -------- Toolkit interne (ta lib) --------
# On importe ce dont on a besoin depuis le toolkit que tu as installé dans le projet
from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig  # sélection de features (profil)
# ⚠️ pipeline_core.py version "temporelle" (celle qu’on vient de mettre à jour)
from t4rec_toolkit.pipeline_core import (
    blank_config,                 # gabarit de configuration
    run_training,                 # pipeline complet (construit séquences + entraîne + sauve)
)

# -------- (Optionnels) pour affichages jolis --------
from datetime import datetime  # timestamps dans les prints

# ============== PARAMÈTRES HAUT NIVEAU (À ADAPTER SI BESOIN) ==============
DATASET_MAIN = "BASE_SCORE_COMPLETE_prepared"     # ← ta table principale de 6M lignes
DATASET_EVENTS = "T4REC_EVENTS_FROM_MAIN"         # ← la vue "événements" (sera créée si absente)
CLIENT_ID_COL = "CLIENT_ID"                       # ← identifiant client
TIME_COL = "DATMAJ"                               # ← colonne temporelle (mensuelle chez toi)
PRODUCT_COL = "SOUSCRIPTION_PRODUIT_1M"           # ← le “produit du mois”
EXTRA_EVENT_COLS = []                             # ← éventuels attributs d’événements (ex: ["CANAL", "FAMILLE"])
PROFILE_CATEGORICAL_CANDIDATES = [                # ← colonnes profil catégorielles utiles (si présentes)
    "MEILLEURE_CARTE_DETENUE_M", "LIBFAMCSP", "CONNAISSANCE_MIF"
]
PROFILE_SEQUENCE_CANDIDATES = [                   # ← colonnes profil numériques (mono-pas) à discrétiser
    "AGE"
]
TARGET_EXCLUDE = ["aucune_souscription"]          # ← exclusion des non-souscriptions côté label
RARE_MIN_COUNT = 200                              # ← <200 clients → classe fusionnée "AUTRES_PRODUITS"

# ============== AFFICHE ENTÊTE ==============
print("PIPELINE T4REC — DIMENSION TEMPORELLE (FS ➜ CONFIG ➜ TRAIN ➜ METRICS)")
print("=" * 80)
print(f"Date/Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset principal          : {DATASET_MAIN}")
print(f"Dataset événements (vue)   : {DATASET_EVENTS}")
print("-" * 80)

# ============== 1) FEATURE SELECTION (FACULTATIF MAIS UTILE) ==============
print("\n1) FEATURE SELECTION SUR LE PROFIL")
print("-" * 80)
try:
    # -- On fait une FS rapide pour repérer quelques features profil pertinentes
    fs_config = FeatureSelectorConfig(
        sample_size=50_000,              # échantillon raisonnable pour aller vite
        total_feature_cap=20,            # budget total de features proposées
        top_k_sequence=12,               # max "séquentielles" (dans ton cas, ce sont des numériques de profil)
        top_k_categorical=8,             # max catégorielles de profil
        compute_model_importance=True,   # importance RF-CV pour trancher
        rf_n_estimators=100,             # taille forêt
        corr_threshold=0.85,             # anti-multicolinéarité
        chunk_size=20_000,               # performance
        downcast_dtypes=True,            # RAM
        correlation_batch_size=100,      # performance
        gc_frequency=50,                 # RAM
        report_dataset=None,             # tu peux mettre un nom de dataset si tu veux sauvegarder le rapport
        verbose=True,
        progress=True,
    )
    print("→ Lancement de la FS sur la table principale…")
    selector = FeatureSelector(fs_config)                               # instancie la FS
    selector.fit(DATASET_MAIN, None, PRODUCT_COL)                       # fit sur la table + target
    selected = selector.get_selected_features()                         # récupère les colonnes choisies
    seq_cols_fs = selected.get("sequence_cols", [])                     # candidates numériques (profil)
    cat_cols_fs = selected.get("categorical_cols", [])                  # candidates catégorielles (profil)

    print(f"   Proposées (num) : {seq_cols_fs}")                         # affichage
    print(f"   Proposées (cat) : {cat_cols_fs}")                         # affichage
except Exception as e:
    print(f"⚠️ FS non bloquante - Erreur: {e}")                           # si la FS échoue, on continue
    seq_cols_fs, cat_cols_fs = [], []                                    # fallback

# -- On croise avec nos “candidats sûrs”, pour rester simple/robuste
profile_categorical_cols = [c for c in PROFILE_CATEGORICAL_CANDIDATES if c in cat_cols_fs or len(cat_cols_fs) == 0]
profile_sequence_cols    = [c for c in PROFILE_SEQUENCE_CANDIDATES if c in seq_cols_fs or len(seq_cols_fs) == 0]

print("\n   → CATEGOS profil retenues :", profile_categorical_cols)      # résumé sélection
print("   → NUMÉRIQUES profil retenues :", profile_sequence_cols)        # résumé sélection

# ============== 2) CONSTRUIRE/MAJ LA VUE “ÉVÉNEMENTS” ==============
print("\n2) CONSTRUCTION DE LA VUE ÉVÉNEMENTS (CLIENT, DATE, PRODUIT)")
print("-" * 80)
try:
    # -- Charger la table principale
    df_main = dataiku.Dataset(DATASET_MAIN).get_dataframe()             # on peut passer limit=… pour tester
    print(f"   Table principale chargée : {df_main.shape}")

    # -- Garder uniquement les colonnes nécessaires (client / temps / produit)
    missing_cols = [c for c in [CLIENT_ID_COL, TIME_COL, PRODUCT_COL] if c not in df_main.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Colonnes requises manquantes dans {DATASET_MAIN} : {missing_cols}")  # garde-fou

    events = df_main[[CLIENT_ID_COL, TIME_COL, PRODUCT_COL] + [c for c in EXTRA_EVENT_COLS if c in df_main.columns]].copy()  # vue étroite

    # -- Nettoyage de base
    events = events.dropna(subset=[CLIENT_ID_COL, TIME_COL])            # supprime lignes sans id ou sans date
    events[TIME_COL] = pd.to_datetime(events[TIME_COL], errors="coerce") # convertit en datetime
    events = events.dropna(subset=[TIME_COL])                           # supprime les dates illisibles

    # -- En cas de multi-lignes (client, mois) : garder la plus récente
    events = (events
              .sort_values([CLIENT_ID_COL, TIME_COL])                   # ordre chrono
              .drop_duplicates(subset=[CLIENT_ID_COL, TIME_COL], keep="last"))  # 1 ligne par (client, mois)

    # -- Écrit/écrase la vue événements dans Dataiku
    dataiku.Dataset(DATASET_EVENTS).write_with_schema(events)           # crée/maj dataset avec schéma auto
    print(f"   ✅ Événements sauvegardés dans {DATASET_EVENTS} : {events.shape}")
except Exception as e:
    # -- Si dataset de sortie n’existe pas encore dans le Flow, Dataiku crée la définition à la volée (projets récents)
    #    Sinon, crée d’abord un dataset vide depuis l’UI, puis relance.
    print(f"❌ Erreur lors de la construction de {DATASET_EVENTS} : {e}")
    raise

# ============== 3) CONFIGURATION DU PIPELINE ==============
print("\n3) CONFIGURATION DU PIPELINE TEMPOREL")
print("-" * 80)
config = blank_config()                                                # récupère un gabarit propre

# -- Données (événements obligatoires)
config["data"]["events_dataset"]         = DATASET_EVENTS              # ← la vue qu’on vient de créer
config["data"]["client_id_col"]          = CLIENT_ID_COL               # ← id client
config["data"]["event_time_col"]         = TIME_COL                    # ← date/mois
config["data"]["product_col"]            = PRODUCT_COL                 # ← item observé (séquence)
config["data"]["event_extra_cols"]       = EXTRA_EVENT_COLS            # ← features d’événements (optionnel)

# -- Données (profil facultatif) — on réutilise la table principale
config["data"]["dataset_name"]           = DATASET_MAIN                # ← source profil
config["data"]["profile_join_key"]       = CLIENT_ID_COL               # ← clé de jointure
config["data"]["profile_categorical_cols"]= profile_categorical_cols   # ← catégorielles profil à répéter sur T
config["data"]["profile_sequence_cols"]  = profile_sequence_cols       # ← numériques profil à discrétiser/répéter

# -- Fenêtre temporelle (séquences)
config["sequence"]["months_lookback"]    = 24                          # ← longueur de séquence (24 mois)
config["sequence"]["time_granularity"]   = "M"                         # ← “M”=mois, “W”=semaine (support de base)
config["sequence"]["min_events_per_client"]= 1                         # ← au moins 1 event pour garder le client
config["sequence"]["target_horizon"]     = 1                           # ← prédire l’item du mois suivant
config["sequence"]["pad_value"]          = 0                           # ← index padding
config["sequence"]["build_target_from_events"] = True                  # ← label dérivé automatiquement

# -- Features / cible
config["features"]["exclude_target_values"] = TARGET_EXCLUDE           # ← on retire “aucune_souscription” des labels
config["features"]["merge_rare_threshold"]  = RARE_MIN_COUNT           # ← classes rares regroupées
config["features"]["other_class_name"]      = "AUTRES_PRODUITS"        # ← nom de la classe fusionnée

# -- Modèle (taille adaptée à ~100k–1M clients)
config["model"]["d_model"]               = 768                         # ← dimension interne
config["model"]["n_heads"]               = 24                          # ← nb têtes attention
config["model"]["n_layers"]              = 8                           # ← profondeur Transformer
config["model"]["dropout"]               = 0.10                        # ← régularisation
config["model"]["max_sequence_length"]   = 24                          # ← doit matcher months_lookback
config["model"]["vocab_size"]            = 2000                        # ← base pour embeddings (sera étendue pour items)

# -- Entraînement
config["training"]["batch_size"]         = 64                          # ← mini-batch
config["training"]["num_epochs"]         = 20                          # ← époques (augmente si temps ok)
config["training"]["learning_rate"]      = 5e-4                        # ← LR AdamW
config["training"]["weight_decay"]       = 1e-4                        # ← régularisation
config["training"]["val_split"]          = 0.20                        # ← split validation
config["training"]["class_weighting"]    = True                        # ← pondération classes (déséquilibre)
config["training"]["gradient_clip"]      = 1.0                         # ← stabilité entraînement
config["training"]["optimizer"]          = "adamw"                     # ← optimiseur

# -- Sorties (datasets Dataiku)
config["outputs"]["features_dataset"]        = "T4REC_FEATURES"        # ← optionnel
config["outputs"]["predictions_dataset"]     = "T4REC_PREDICTIONS"     # ← optionnel
config["outputs"]["metrics_dataset"]         = "T4REC_METRICS"         # ← optionnel
config["outputs"]["model_artifacts_dataset"] = "T4REC_MODEL"           # ← optionnel
config["outputs"]["local_dir"]               = "output"                # ← répertoire local (logs/exports)

# -- Runtime
config["runtime"]["verbose"]             = True                        # ← logs détaillés
config["runtime"]["progress"]            = True                        # ← barre de progression
config["runtime"]["seed"]                = 42                          # ← reproductibilité

# -- Affichage rapide de l’archi
print(f"   Archi    : {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D")
print(f"   Séquence : {config['sequence']['months_lookback']} mois (horizon={config['sequence']['target_horizon']})")
print(f"   Profil   : cat={config['data']['profile_categorical_cols']} | num={config['data']['profile_sequence_cols']}")
print(f"   Exclu y  : {config['features']['exclude_target_values']}")
print(f"   Rares    : <{config['features']['merge_rare_threshold']} → '{config['features']['other_class_name']}'")

# ============== 4) ENTRAÎNEMENT ==============
print("\n4) ENTRAÎNEMENT DU MODÈLE")
print("-" * 80)
try:
    results = run_training(config)                                     # ← lance tout le pipeline (séquences + training)
    print("   ✅ Entraînement terminé.")
except Exception as e:
    print(f"❌ ERREUR durant l'entraînement: {e}")                      # ← capture l’erreur pour debug rapide
    raise

# ============== 5) RÉSULTATS & MÉTRIQUES ==============
print("\n5) RÉSULTATS & MÉTRIQUES")
print("-" * 80)

# -- Métriques standard
metrics = results.get("metrics", {})                                   # ← dict accuracy/precision/recall/f1
print("   → Métriques (validation) :")
for k in ["accuracy", "precision", "recall", "f1"]:
    v = metrics.get(k, None)
    if v is not None:
        print(f"     - {k:9s} : {v:.4f}")

# -- Info modèle / données
mi = results.get("model_info", {})
di = results.get("data_info", {})
print("\n   → Modèle : ", mi.get("architecture", "N/A"))
print("   → Paramètres (≈) :", f"{mi.get('total_params', 0):,}")
print("   → Données :")
print(f"     - Clients validés : {di.get('n_clients', 'N/A')}")
print(f"     - Longueur séq.   : {di.get('seq_len', 'N/A')}")
print(f"     - #Classes cible  : {di.get('n_classes', 'N/A')}")
print(f"     - Features (emb)  : {di.get('features', [])}")

# -- Datasets sauvegardés
saved = results.get("saved_datasets", {})
if saved:
    print("\n   → Datasets Dataiku créés/sauvegardés :")
    for k, v in saved.items():
        print(f"     - {k:12s}: {v}")
else:
    print("\n   → Aucun dataset Dataiku sauvegardé (sorties désactivées ou non déclarées).")

# -- (Option) Lire les métriques Top-K depuis le dataset (si activé)
#    Le pipeline les a déjà calculées et écrit “T4REC_METRICS” si configuré.
try:
    if config["outputs"]["metrics_dataset"]:
        met_df = dataiku.Dataset(config["outputs"]["metrics_dataset"]).get_dataframe()
        # On filtre les métriques top-k “nbo” pour un aperçu rapide
        topk_view = (met_df[(met_df["metric_type"] == "topk_nbo")]
                          .sort_values(["k_value", "metric_name"]))
        print("\n   → Aperçu métriques Top-K (depuis dataset) :")
        # Affiche quelques lignes (tu peux passer à .head(20))
        print(topk_view.head(20))
except Exception as e:
    print(f"⚠️ Impossible de relire les métriques top-k depuis Dataiku: {e}")

print("\n" + "=" * 80)
print("PIPELINE TERMINÉ ✅")
