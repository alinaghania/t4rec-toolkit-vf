
# =============================================================================
# Imports
# =============================================================================

import sys
import traceback
import numpy as np
import pandas as pd

# Dataiku (si notebook exécuté dans Dataiku)
try:
    import dataiku
    _HAS_DATAIKU = True
except Exception:
    _HAS_DATAIKU = False

# Import du pipeline hybride et des utilitaires Top-K
from t4rec_toolkit.pipeline_core import (
    blank_config,
    validate_config,
    run_training,
    evaluate_topk_metrics_nbo,
    print_topk_results,
)

print("PIPELINE T4REC HYBRIDE - DÉMARRAGE")
print("=" * 80)

# =============================================================================
# (Optionnel) Import FeatureSelector si disponible
# =============================================================================
# La sélection de features est utile quand tu as 100+ colonnes.
# Si ton utilitaire n'est pas encore prêt, on tombera sur un fallback simple.

_HAS_FS = False
try:
    from t4rec_toolkit.utils import FeatureSelector, FeatureSelectorConfig
    _HAS_FS = True
    print("FeatureSelector détecté. La sélection intelligente de features sera utilisée.")
except Exception:
    print("FeatureSelector non disponible → fallback (sélection manuelle / heuristique).")

# =============================================================================
# 1) FEATURE SELECTION (facultative) / OU LISTE MANUELLE
# =============================================================================

print("\n1. FEATURE SELECTION / LISTE MANUELLE")
print("-" * 80)

DATASET_NAME = "BASE_SCORE_COMPLETE_prepared"  # ← adapte si besoin
TARGET_COL   = "SOUSCRIPTION_PRODUIT_1M"      # ← adapte si besoin
EXCLUDE_TARGET_VALUES = ["aucune_souscription"]  # libellé exact à exclure (minuscule/majuscules indifférent)

selected_features = {
    "sequence_cols": [],
    "categorical_cols": [],
}

if _HAS_FS:
    # --- Config de la sélection de features ---
    fs_config = FeatureSelectorConfig(
        sample_size=30000,             # sous-échantillon pour la FS
        total_feature_cap=20,          # nombre total max de features
        top_k_sequence=12,             # ratio séquentielles
        top_k_categorical=8,           # ratio catégorielles
        compute_model_importance=True,
        rf_n_estimators=100,
        corr_threshold=0.85,
        chunk_size=15000,
        downcast_dtypes=True,
        correlation_batch_size=100,
        gc_frequency=50,
        report_dataset="T4REC_FEATURE_REPORT_100K",  # dataset de reporting
        verbose=True,
        progress=True,
    )

    print("Lancement de la sélection de features...")
    selector = FeatureSelector(fs_config)
    selector.fit(DATASET_NAME, None, TARGET_COL)
    selected_features = selector.get_selected_features()

    print(
        f" Sélectionné: {len(selected_features['sequence_cols'])} séquentielles + "
        f"{len(selected_features['categorical_cols'])} catégorielles"
    )
    print(f"Séquentielles: {selected_features['sequence_cols']}")
    print(f"Catégorielles: {selected_features['categorical_cols']}")
else:
    # --- Fallback si tu n'as pas encore la FeatureSelector ---
    # 1) Charger un head du dataset pour détecter automatiquement quelques colonnes plausibles
    #    (ou tu peux lister toi-même ici si tu sais déjà lesquelles utiliser)
    if _HAS_DATAIKU:
        print("Fallback FS : détection heuristique des colonnes (échantillon du dataset).")
        df_head = dataiku.Dataset(DATASET_NAME).get_dataframe(limit=2000)
    else:
        raise RuntimeError("Ce notebook s'attend à être exécuté dans Dataiku pour le fallback FS.")

    # Heuristique simple :
    # - colonnes numériques → candidates "séquentielles" (en réalité scalaires → seq_len=1)
    # - colonnes object/catégorie (basse cardinalité) → catégorielles
    seq_candidates = []
    cat_candidates = []

    for col in df_head.columns:
        if col == TARGET_COL:
            continue
        s = df_head[col]
        if pd.api.types.is_numeric_dtype(s):
            # garder si non-constante
            if s.nunique(dropna=True) > 5:
                seq_candidates.append(col)
        elif s.dtype == "object" or s.dtype.name == "category":
            nunq = s.nunique(dropna=True)
            if 2 <= nunq <= 50:
                cat_candidates.append(col)

    # Limiter pour ne pas exploser la dimension d'embedding
    selected_features["sequence_cols"] = seq_candidates[:12]   # pick top 12 heuristiques
    selected_features["categorical_cols"] = cat_candidates[:8] # pick top 8 heuristiques

    print("Sélection heuristique OK.")
    print(
        f" Sélectionné: {len(selected_features['sequence_cols'])} séquentielles + "
        f"{len(selected_features['categorical_cols'])} catégorielles"
    )
    print(f"Séquentielles: {selected_features['sequence_cols']}")
    print(f"Catégorielles: {selected_features['categorical_cols']}")


# Sécurité minimum: si on n'a rien trouvé, on stoppe proprement
if not selected_features["sequence_cols"] and not selected_features["categorical_cols"]:
    raise RuntimeError("Aucune feature sélectionnée. Fournis au moins 1 col séquentielle ou catégorielle.")

# =============================================================================
# 2) CONFIGURATION T4REC HYBRIDE (customisable)
# =============================================================================

print("\n2. CONFIGURATION T4REC HYBRIDE")
print("-" * 80)

config = blank_config()

# Données
config["data"]["dataset_name"] = DATASET_NAME
config["data"]["sample_size"] = 1000000        # nb de lignes max à charger (limite Dataiku/get_dataframe)
config["data"]["chunk_size"] = 400000          # non vital ici (optim batch I/O)
config["data"]["limit"] = None
config["data"]["partitions"] = None            # éviter soucis de partition si non utilisées
config["data"]["temporal_split"] = None        # split train/val classique (stratifié)

# Features (issues de la FS ou du fallback)
config["features"]["sequence_cols"] = selected_features["sequence_cols"]
config["features"]["categorical_cols"] = selected_features["categorical_cols"]
config["features"]["target_col"] = TARGET_COL
config["features"]["exclude_target_values"] = EXCLUDE_TARGET_VALUES

# 🧠 Modèle — IMPORTANT: max_sequence_length=1 tant qu'on ne construit pas de vraies séquences temporelles
config["model"]["d_model"] = 512          # dimension interne Transformer
config["model"]["n_heads"] = 8            # doit diviser d_model
config["model"]["n_layers"] = 4           # profondeur
config["model"]["dropout"] = 0.1
config["model"]["max_sequence_length"] = 1   # ⚠️ on reste à 1 pour l’instant
config["model"]["mem_len"] = 100
config["model"]["attn_type"] = "bi"
config["model"]["vocab_size"] = 2000      # suffisamment grand pour mapper indices

# Entraînement
config["training"]["batch_size"] = 128
config["training"]["num_epochs"] = 20
config["training"]["learning_rate"] = 5e-4
config["training"]["weight_decay"] = 5e-4
config["training"]["gradient_clip"] = 0.5
config["training"]["val_split"] = 0.2
config["training"]["optimizer"] = "adamw"
config["training"]["scheduler"] = "cosine"      # ou None
config["training"]["warmup_steps"] = 0
config["training"]["early_stopping_patience"] = 0  # 0 = off

# Métriques standards
config["metrics"] = ["accuracy", "precision", "recall", "f1"]

# Sorties Dataiku (datasets à créer en OUTPUT du recipe/notebook si tu veux les écrire)
config["outputs"]["features_dataset"] = "T4REC_FEATURES"
config["outputs"]["predictions_dataset"] = "T4REC_PREDICTIONS"
config["outputs"]["metrics_dataset"] = "T4REC_METRICS"
config["outputs"]["model_artifacts_dataset"] = "T4REC_MODEL"
config["outputs"]["local_dir"] = "output"
config["outputs"]["save_model"] = True

# Runtime
config["runtime"]["verbose"] = True
config["runtime"]["progress"] = True
config["runtime"]["seed"] = 42
config["runtime"]["memory_efficient"] = True
config["runtime"]["checkpoint_freq"] = 0  # non utilisé ici

print(
    f"Architecture: {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D"
)
print(
    f"Training: {config['training']['num_epochs']} epochs, LR={config['training']['learning_rate']}"
)
print("Outputs prévus (Dataiku):", {k:v for k,v in config["outputs"].items() if v and k.endswith('_dataset')})

# =============================================================================
# 3) VALIDATION CONFIG & ENTRAÎNEMENT
# =============================================================================

print("\n3. VALIDATION & ENTRAÎNEMENT")
print("-" * 80)

errors = validate_config(config)
if errors:
    print(" ERREURS DE CONFIGURATION:")
    for err in errors:
        print(f"  - {err}")
    raise SystemExit("Arrêt (config invalide).")
else:
    print(" Configuration validée. Lancement entraînement...")

try:
    results = run_training(config)

    print("\n" + "=" * 80)
    print("RÉSULTATS ENTRAÎNEMENT T4REC HYBRIDE")
    print("=" * 80)

    # --- Métriques finales
    print("\n MÉTRIQUES FINALES (validation):")
    for metric, value in results["metrics"].items():
        if isinstance(value, (int, float, np.floating)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # --- Infos modèle
    print("\n INFOS MODÈLE:")
    mi = results["model_info"]
    print(f"  Paramètres: {mi.get('total_params', 'N/A'):,}")
    print(f"  Architecture: {mi.get('architecture', 'N/A')}")

    # --- Infos données
    print("\n INFOS DONNÉES:")
    di = results["data_info"]
    def fmt_int(x): 
        return f"{x:,}" if isinstance(x, (int, np.integer)) else str(x)
    print(f"  Lignes chargées: {fmt_int(di.get('rows','N/A'))}")
    print(f"  Train: {fmt_int(di.get('train_samples','N/A'))} | Val: {fmt_int(di.get('val_samples','N/A'))}")
    print(f"  Features: {fmt_int(di.get('n_features','N/A'))} "
          f"(seq={fmt_int(di.get('n_sequence_features','N/A'))}, cat={fmt_int(di.get('n_categorical_features','N/A'))})")
    print(f"  #Classes: {fmt_int(di.get('target_classes','N/A'))}")

    # =============================================================================
    # 4) TOP-K (format collègue)
    # =============================================================================
    print("\n4. ÉVALUATION TOP-K (format collègue)")
    print("-" * 80)

    if "predictions" in results:
        preds = results["predictions"]
        raw_probs = preds["raw_outputs"]    # ← probabilités (softmax) [N, C]
        true_cls  = preds["true_classes"]   # ← entiers [N]

        print(f"  Nb prédictions: {len(raw_probs):,}")
        k_values = [1, 3, 5]

        # **Inverse mapping des classes vers labels d'origine**
        # On relit un petit échantillon du dataset pour reconstituer un LabelEncoder
        # identique (même ordre de classes) → mapping indice -> libellé produit.
        if _HAS_DATAIKU:
            # pour minimiser le coût, on ne lit qu'un sous-échantillon
            df_small = dataiku.Dataset(DATASET_NAME).get_dataframe(limit=50000)
            # Nettoyage des valeurs exclues
            if EXCLUDE_TARGET_VALUES:
                mask_excl = df_small[TARGET_COL].astype(str).str.lower().isin([v.lower() for v in EXCLUDE_TARGET_VALUES])
                df_small = df_small.loc[~mask_excl].reset_index(drop=True)
            # Fit LabelEncoder sur les valeurs présentes (ordre identique si le set est représentatif)
            from sklearn.preprocessing import LabelEncoder
            le_tmp = LabelEncoder()
            le_tmp.fit(df_small[TARGET_COL])
            inverse_target_mapping = {i: str(c) for i, c in enumerate(le_tmp.classes_)}
        else:
            # fallback: nom générique
            n_classes = raw_probs.shape[1]
            inverse_target_mapping = {i: f"CLASS_{i}" for i in range(n_classes)}

        # Calcul Top-K façon collègue
        topk_metrics = evaluate_topk_metrics_nbo(
            predictions=raw_probs,
            targets=true_cls,
            inverse_target_mapping=inverse_target_mapping,
            k_values=k_values,
        )

        print("\n📊 MÉTRIQUES TOP-K (code collègue):")
        for k in k_values:
            if k in topk_metrics:
                m = topk_metrics[k]
                print(f"\n  K={k} (sur {m.get('Clients_evaluated', 0)} clients valides):")
                print(f"    Precision@{k}: {m.get('Precision@K', 0):.4f}")
                print(f"    Recall@{k}:    {m.get('Recall@K', 0):.4f}")
                print(f"    F1@{k}:        {m.get('F1@K', 0):.4f}   (NB: peut être approximatif)")
                print(f"    NDCG@{k}:      {m.get('NDCG@K', 0):.4f}")
                print(f"    MAP:           {m.get('MAP', 0):.4f}")
                print(f"    HitRate@{k}:   {m.get('HitRate@K', 0):.4f}")
                print(f"    Coverage@{k}:  {m.get('Coverage@K', 0):.4f}")
        # Affichage sous forme de tableau lisible
        print("\n" + "-" * 80)
        print_topk_results(topk_metrics)

    else:
        print("Pas de données de prédiction disponibles dans results['predictions'].")

    # =============================================================================
    # 5) DATASETS SAUVÉS (Dataiku)
    # =============================================================================
    print("\n5. DATASETS SAUVEGARDÉS")
    print("-" * 80)

    saved = results.get("saved_datasets", {})
    if saved:
        print(" SUCCÈS - Datasets écrits via Dataiku:")
        for key, name in saved.items():
            print(f"   - {key:<15}: {name}")
    else:
        print(" Aucun dataset n'a été sauvegardé.")
        print("   → Vérifie que les datasets OUTPUT sont définis côté recipe / notebook")
        print("   → Et que tu as les droits d'écriture")

    # =============================================================================
    # RÉSUMÉ FINAL
    # =============================================================================
    print("\n" + "=" * 80)
    print("RÉSUMÉ PIPELINE COMPLET")
    print("=" * 80)
    print(f" Données: {fmt_int(results['data_info'].get('rows','N/A'))} lignes (split stratifié)")
    print(f" Features utilisées: {len(config['features']['sequence_cols']) + len(config['features']['categorical_cols'])}")
    print(f" Modèle: {results['model_info'].get('architecture','N/A')}")
    print(f" Validation: {fmt_int(results['data_info'].get('val_samples','N/A'))} échantillons")
    print(f" Datasets écrits: {saved if saved else 'aucun'}")

except Exception as e:
    print("\n" + "!" * 80)
    print("ERREUR DURANT L'ENTRAÎNEMENT / ÉVALUATION")
    print("Message:", str(e))
    traceback.print_exc(file=sys.stdout)
    print("!" * 80)

print("\nPIPELINE TERMINÉ ✅")
