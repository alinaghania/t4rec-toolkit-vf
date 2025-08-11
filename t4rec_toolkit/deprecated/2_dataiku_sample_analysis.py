# === ANALYSE RAPIDE ÉCHANTILLON - BASE_SCORE_COMPLETE_prepared ===
# Version optimisée pour analyser rapidement la structure sans surcharger

import dataiku
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

print("🔍 ANALYSE RAPIDE ÉCHANTILLON - BASE_SCORE_COMPLETE_prepared")
print("=" * 65)

# === CONFIGURATION ÉCHANTILLONNAGE ===
SAMPLE_SIZE = 1000  # Petit échantillon pour rapidité
MAX_PARTITIONS_TO_CHECK = 2  # Maximum 2 partitions pour aller vite

# === CONNEXION DATASETS DATAIKU ===
print("\n📊 1. CONNEXION DATASET...")

# Input : Table partitionnée source
input_dataset = dataiku.Dataset("BASE_SCORE_COMPLETE_prepared")

# Output : Résultats d'analyse rapide
output_dataset = dataiku.Dataset("T4REC_SAMPLE_ANALYSIS")

# === RÉCUPÉRATION ÉCHANTILLON RAPIDE ===
print(f"\n🔬 2. ÉCHANTILLONNAGE RAPIDE ({SAMPLE_SIZE} lignes)...")


def get_quick_sample():
    """Récupère rapidement un échantillon représentatif"""
    try:
        # Méthode 1 : Essayer de prendre directement un échantillon
        print("   📊 Tentative échantillon direct...")
        df_sample = input_dataset.get_dataframe(limit=SAMPLE_SIZE)
        print(f"   ✅ Échantillon obtenu: {df_sample.shape}")
        return df_sample, "direct"

    except Exception as e:
        print(f"   ⚠️ Échantillon direct échoué: {e}")

        try:
            # Méthode 2 : Essayer via une partition récente
            print("   📊 Tentative via partitions...")
            partitions = input_dataset.list_partitions()

            # Prendre les partitions les plus récentes (probablement 2024)
            recent_partitions = sorted(partitions)[-MAX_PARTITIONS_TO_CHECK:]
            print(f"   📅 Partitions testées: {recent_partitions}")

            for partition in recent_partitions:
                try:
                    df_sample = input_dataset.get_dataframe(
                        partition=partition, limit=SAMPLE_SIZE
                    )
                    if len(df_sample) > 0:
                        print(
                            f"   ✅ Échantillon obtenu via {partition}: {df_sample.shape}"
                        )
                        return df_sample, f"partition_{partition}"
                except Exception as pe:
                    print(f"   ⚠️ Partition {partition} échouée: {pe}")
                    continue

            # Si aucune partition ne marche
            print("   ❌ Aucune partition accessible")
            return None, "failed"

        except Exception as e2:
            print(f"   ❌ Méthode partitions échouée: {e2}")
            return None, "failed"


# Récupérer l'échantillon
df_sample, method = get_quick_sample()

if df_sample is None:
    print("❌ IMPOSSIBLE D'OBTENIR UN ÉCHANTILLON")
    print("🔍 Vérifiez les permissions d'accès au dataset")
    exit()

print(f"✅ Échantillon récupéré via: {method}")

# === ANALYSE RAPIDE STRUCTURE ===
print(f"\n🏗️ 3. ANALYSE STRUCTURE ({df_sample.shape})...")

# Infos de base
total_columns = len(df_sample.columns)
total_rows = len(df_sample)
memory_mb = df_sample.memory_usage(deep=True).sum() / 1024 / 1024

print(f"📊 Dimensions: {total_rows:,} lignes × {total_columns} colonnes")
print(f"💾 Mémoire échantillon: {memory_mb:.1f} MB")

# === COMPARAISON AVEC TEST PRÉCÉDENT ===
print("\n🔍 4. COMPATIBILITÉ AVEC VOTRE TEST...")

# Colonnes attendues du test pipeline_t4rec_training.py
expected_columns = {
    "sequences": [
        "nbchqemigliss_m12",
        "nb_automobile_12dm",
        "mntecscrdimm",
        "mnt_euro_r_3m",
        "nb_contacts_accueil_service",
    ],
    "categoriques": ["dummy:iac_epa:03", "dummy:iac_epa:01", "dummy:iac_epa:02"],
    "target": "souscription_produit_1m",
}

all_columns = list(df_sample.columns)
compatibility_report = {}

# Vérifier chaque type de colonne
for col_type, col_list in expected_columns.items():
    if col_type == "target":
        # Target est une seule colonne
        found = col_list in all_columns
        compatibility_report[col_type] = {
            "expected": col_list,
            "found": found,
            "missing": [] if found else [col_list],
        }
        print(f"🎯 Target '{col_list}': {'✅ TROUVÉE' if found else '❌ MANQUANTE'}")
    else:
        # Listes de colonnes
        missing = [col for col in col_list if col not in all_columns]
        found_count = len(col_list) - len(missing)
        compatibility_report[col_type] = {
            "expected": col_list,
            "found_count": found_count,
            "missing": missing,
        }
        print(f"📋 {col_type.capitalize()}: {found_count}/{len(col_list)} trouvées")
        if missing:
            print(f"   ❌ Manquantes: {missing}")

# === RECHERCHE COLONNES SIMILAIRES ===
print("\n🔍 5. RECHERCHE COLONNES SIMILAIRES...")


def find_similar_columns(missing_col, all_cols, max_results=3):
    """Trouve des colonnes similaires par nom"""
    similar = []
    missing_lower = missing_col.lower()

    # Recherche par mots-clés
    keywords = missing_lower.replace(":", "_").split("_")

    for col in all_cols:
        col_lower = col.lower()
        # Si au moins 2 mots-clés correspondent
        matches = sum(1 for kw in keywords if kw in col_lower and len(kw) > 2)
        if matches >= 2:
            similar.append(col)

    return similar[:max_results]


similar_suggestions = {}
all_missing = []

for col_type, report in compatibility_report.items():
    if "missing" in report and report["missing"]:
        all_missing.extend(report["missing"])

for missing_col in all_missing:
    similar = find_similar_columns(missing_col, all_columns)
    if similar:
        similar_suggestions[missing_col] = similar
        print(f"💡 '{missing_col}' → Similaires: {similar}")

# === ANALYSE QUALITÉ RAPIDE ===
print("\n📊 6. QUALITÉ DES DONNÉES (TOP 20 colonnes)...")

quality_summary = {}
sample_columns = all_columns[:20]  # Analyser seulement les 20 premières pour rapidité

for col in sample_columns:
    try:
        null_pct = round((df_sample[col].isnull().sum() / len(df_sample)) * 100, 1)
        unique_count = df_sample[col].nunique()
        dtype = str(df_sample[col].dtype)

        quality_summary[col] = {
            "null_percentage": null_pct,
            "unique_values": unique_count,
            "dtype": dtype,
        }

        # Affichage condensé
        status = "🟢" if null_pct < 10 else "🟡" if null_pct < 50 else "🔴"
        print(
            f"   {status} {col[:30]:<30} | {null_pct:>5.1f}% null | {unique_count:>4} uniques | {dtype}"
        )

    except Exception as e:
        quality_summary[col] = {"error": str(e)}

# === ESTIMATION VOLUMÉTRIE RAPIDE ===
print("\n📈 7. ESTIMATION VOLUMÉTRIE 2024...")

# Estimation très approximative basée sur l'échantillon
if method.startswith("partition"):
    # Si on a une partition, estimer pour 12 mois
    estimated_rows_2024 = total_rows * 12
elif method == "direct":
    # Si échantillon direct, assumer que c'est représentatif
    full_dataset_estimate = total_rows * 100  # Facteur approximatif
    estimated_rows_2024 = full_dataset_estimate

estimated_memory_gb = (estimated_rows_2024 * total_columns * 8) / (1024**3)
estimated_processing_hours = estimated_rows_2024 / 10000  # Très approximatif

print(f"📊 Estimation 2024:")
print(f"   📈 Lignes estimées: ~{estimated_rows_2024:,}")
print(f"   💾 Mémoire estimée: ~{estimated_memory_gb:.1f} GB")
print(f"   ⏱️ Temps traitement estimé: ~{estimated_processing_hours:.1f}h")

# === RECOMMANDATIONS RAPIDES ===
print("\n💡 8. RECOMMANDATIONS...")

recommendations = []

# Compatibilité
target_ok = compatibility_report.get("target", {}).get("found", False)
seq_missing = len(compatibility_report.get("sequences", {}).get("missing", []))
cat_missing = len(compatibility_report.get("categoriques", {}).get("missing", []))

if not target_ok:
    recommendations.append(
        "❌ CRITIQUE: Target manquante - Pipeline non utilisable tel quel"
    )
if seq_missing > 2:
    recommendations.append(
        f"⚠️ {seq_missing} colonnes séquentielles manquantes - Adapter le pipeline"
    )
if cat_missing > 1:
    recommendations.append(
        f"⚠️ {cat_missing} colonnes catégorielles manquantes - Vérifier encodage"
    )

# Volume
if estimated_memory_gb > 32:
    recommendations.append(
        f"💾 Volume important ({estimated_memory_gb:.1f}GB) - Processing par chunks nécessaire"
    )
if estimated_processing_hours > 4:
    recommendations.append(
        f"⏱️ Traitement long ({estimated_processing_hours:.1f}h) - Considérer sous-échantillonnage"
    )

# Qualité
high_null_cols = [
    col
    for col, info in quality_summary.items()
    if isinstance(info, dict) and info.get("null_percentage", 0) > 50
]
if high_null_cols:
    recommendations.append(
        f"🔴 {len(high_null_cols)} colonnes >50% nulls - Vérifier qualité"
    )

# Afficher recommandations
for i, rec in enumerate(recommendations, 1):
    print(f"   {i}. {rec}")

# === VERDICT FINAL ===
print("\n" + "=" * 65)
print("🎯 VERDICT ANALYSE RAPIDE")
print("=" * 65)

# Score de compatibilité
compatibility_score = 0
if target_ok:
    compatibility_score += 40
compatibility_score += max(0, (5 - seq_missing) * 8)  # 8 points par colonne seq trouvée
compatibility_score += max(0, (3 - cat_missing) * 4)  # 4 points par colonne cat trouvée

if compatibility_score >= 80:
    verdict = "🟢 EXCELLENT - Pipeline directement applicable"
elif compatibility_score >= 60:
    verdict = "🟡 BON - Adaptations mineures nécessaires"
elif compatibility_score >= 40:
    verdict = "🟠 MOYEN - Adaptations importantes nécessaires"
else:
    verdict = "🔴 PROBLÉMATIQUE - Refonte majeure nécessaire"

print(f"📊 Score compatibilité: {compatibility_score}/100")
print(f"🎯 Verdict: {verdict}")

# === SAUVEGARDE RÉSULTATS RAPIDES ===
print(f"\n💾 9. SAUVEGARDE RÉSULTATS...")

# Créer un DataFrame de résultats condensé
results_data = []

# Résumé global
results_data.append(
    {
        "metric_type": "SUMMARY",
        "metric_name": "sample_size",
        "value": total_rows,
        "details": f"Columns: {total_columns}, Memory: {memory_mb:.1f}MB, Method: {method}",
    }
)

results_data.append(
    {
        "metric_type": "COMPATIBILITY",
        "metric_name": "compatibility_score",
        "value": compatibility_score,
        "details": verdict,
    }
)

# Colonnes manquantes
for missing_col in all_missing:
    results_data.append(
        {
            "metric_type": "MISSING_COLUMN",
            "metric_name": missing_col,
            "value": 0,
            "details": json.dumps(similar_suggestions.get(missing_col, [])),
        }
    )

# Recommandations
for i, rec in enumerate(recommendations):
    results_data.append(
        {
            "metric_type": "RECOMMENDATION",
            "metric_name": f"recommendation_{i + 1}",
            "value": 1,
            "details": rec,
        }
    )

# Estimations
results_data.append(
    {
        "metric_type": "ESTIMATION",
        "metric_name": "estimated_rows_2024",
        "value": estimated_rows_2024,
        "details": f"Memory: {estimated_memory_gb:.1f}GB, Time: {estimated_processing_hours:.1f}h",
    }
)

# Créer DataFrame
df_results = pd.DataFrame(results_data)
df_results["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_results["ready_for_pipeline"] = compatibility_score >= 60

print(f"✅ {len(df_results)} résultats préparés")

# Sauvegarder
try:
    output_dataset.write_with_schema(df_results)
    print("✅ Résultats sauvegardés dans T4REC_SAMPLE_ANALYSIS")
except Exception as e:
    print(f"⚠️ Erreur sauvegarde: {e}")
    print("📊 Aperçu résultats:")
    print(df_results[["metric_type", "metric_name", "value"]].head(10))

print("\n" + "=" * 65)
if compatibility_score >= 60:
    print("🚀 PRÊT POUR PHASE 2 - Adaptation pipeline T4Rec")
else:
    print("🔧 CORRECTIONS NÉCESSAIRES avant Phase 2")
print("=" * 65)
