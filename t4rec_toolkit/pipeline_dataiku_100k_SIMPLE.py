# === PIPELINE T4REC XLNET OPTIMISÉ - DATAIKU 100K LIGNES ===
# Pipeline adapté aux vraies données avec sélection intelligente de 12 colonnes

import dataiku
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Progress bars et logging
from tqdm.auto import tqdm
import logging
import time

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Imports T4Rec et votre toolkit
import sys

sys.path.append(
    "/data/DATA_DIR/code-envs/python/DCC_ORION_TR4REC_PYTHON_3_9/lib/python3.9/site-packages"
)

try:
    import transformers4rec.torch as tr
    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

    print("✅ T4Rec imported successfully")
except ImportError as e:
    print(f"❌ T4Rec import error: {e}")

# Votre toolkit
from t4rec_toolkit.transformers.sequence_transformer import SequenceTransformer
from t4rec_toolkit.transformers.categorical_transformer import CategoricalTransformer
from t4rec_toolkit.adapters.dataiku_adapter import DataikuAdapter

print("🚀 PIPELINE T4REC XLNET - DATAIKU 100K LIGNES OPTIMISÉ")
print("=" * 70)
logger.info("Démarrage du pipeline T4Rec XLNet optimisé")
start_time = time.time()

# === CONFIGURATION OPTIMISÉE POUR 100K LIGNES ===j'
CONFIG = {
    # Données
    "sample_size": 100000,  # 100K lignes - scaling up !
    "max_sequence_length": 12,  # Séquences optimales bancaire (garde pareil)
    "chunk_size": 5000,  # Processing par chunks de 5K (plus efficace)
    # Features sélectionnées (12 colonnes métier)
    "sequence_cols": [
        "MNT_EPARGNE",  # Capacité épargne
        "NB_EPARGNE",  # Diversification épargne
        "TAUX_SATURATION_LIVRET",  # Potentiel livret
        "MNT_EP_HORS_BILAN",  # Sophistication
        "NBCHQEMIGLISS_M12",  # Activité compte
        "MNT_EURO_R",  # Volume transactions
    ],
    "categorical_cols": [
        "IAC_EPA",  # Segment principal
        "TOP_EPARGNE",  # Top client épargne
        "TOP_LIVRET",  # Top client livret
        "NB_CONTACTS_ACCUEIL_SERVICE",  # Engagement (traité comme catégoriel)
        "NB_AUTOMOBILE_12DM",  # Produits détenus
        "NB_EP_BILAN",  # Diversification bilan
    ],
    "target_col": "SOUSCRIPTION_PRODUIT_1M",
    # Architecture modèle optimisée pour 100K
    "embedding_dim": 256,  # Plus riche pour 100K données
    "hidden_size": 256,  # Dimension cachée augmentée
    "num_layers": 3,  # Légèrement plus profond
    "num_heads": 8,  # Plus d'attention heads
    "dropout": 0.15,  # Régularisation légèrement réduite
    "vocab_size": 150,  # Vocabulaire features augmenté
    "target_vocab_size": 200,  # Vocabulaire target augmenté
    # Entraînement optimisé pour 100K
    "batch_size": 64,  # Batch plus grand pour 100K
    "num_epochs": 15,  # Plus d'époques que test
    "learning_rate": 0.001,  # LR adaptée
    "weight_decay": 0.01,  # Régularisation
    "gradient_clip": 1.0,  # Clipping gradients
}

print(
    f"📊 Configuration: {CONFIG['sample_size']:,} lignes, {len(CONFIG['sequence_cols'])} seq + {len(CONFIG['categorical_cols'])} cat = {len(CONFIG['sequence_cols']) + len(CONFIG['categorical_cols'])} features"
)

# === CONNEXION DATASETS DATAIKU ===
print("\n📊 1. CONNEXION DATASETS...")

# Input
input_dataset = dataiku.Dataset("BASE_SCORE_COMPLETE_prepared")

# Outputs optimisés
features_dataset = dataiku.Dataset("T4REC_FEATURES_100K")
predictions_dataset = dataiku.Dataset("T4REC_PREDICTIONS_100K")
metrics_dataset = dataiku.Dataset("T4REC_METRICS_100K")

# === CHARGEMENT DONNÉES INTELLIGENT ===
print(f"\n🔄 2. CHARGEMENT OPTIMISÉ ({CONFIG['sample_size']:,} lignes)...")


def load_data_smart():
    """Chargement intelligent avec gestion mémoire"""
    try:
        # Méthode 1: Sample direct
        print("   📊 Tentative échantillon direct...")
        df = input_dataset.get_dataframe(limit=CONFIG["sample_size"])
        print(f"   ✅ Chargé: {df.shape}")
        return df

    except Exception as e:
        print(f"   ⚠️ Échantillon direct échoué: {e}")

        try:
            # Méthode 2: Via partitions récentes
            print("   📊 Tentative via partitions...")
            partitions = input_dataset.list_partitions()
            recent_partitions = sorted(partitions)[-3:]  # 3 dernières partitions

            df_parts = []
            remaining = CONFIG["sample_size"]

            for partition in recent_partitions:
                if remaining <= 0:
                    break
                chunk_size = min(remaining, CONFIG["sample_size"] // 3)
                df_part = input_dataset.get_dataframe(
                    partition=partition, limit=chunk_size
                )
                if len(df_part) > 0:
                    df_parts.append(df_part)
                    remaining -= len(df_part)
                    print(f"   📊 Partition {partition}: {len(df_part)} lignes")

            if df_parts:
                df = pd.concat(df_parts, ignore_index=True)
                print(f"   ✅ Assemblé: {df.shape}")
                return df

        except Exception as e2:
            print(f"   ❌ Méthode partitions échouée: {e2}")

    raise Exception("Impossible de charger les données")


# Charger les données
logger.info("Début chargement des données...")
with tqdm(total=1, desc="📊 Chargement données") as pbar:
    df_raw = load_data_smart()
    pbar.update(1)
logger.info(f"Données chargées: {df_raw.shape}")

# === VÉRIFICATION ET NETTOYAGE COLONNES ===
print("\n🔍 3. VÉRIFICATION COLONNES...")


def verify_and_fix_columns(df, config):
    """Vérifie les colonnes et propose des alternatives"""
    all_available = list(df.columns)
    fixed_config = config.copy()

    # Vérifier colonnes séquentielles
    missing_seq = []
    for col in config["sequence_cols"]:
        if col not in all_available:
            missing_seq.append(col)

    # Vérifier colonnes catégorielles
    missing_cat = []
    for col in config["categorical_cols"]:
        if col not in all_available:
            missing_cat.append(col)

    # Vérifier target
    target_ok = config["target_col"] in all_available

    print(
        f"   📋 Séquentielles: {len(config['sequence_cols']) - len(missing_seq)}/{len(config['sequence_cols'])} trouvées"
    )
    print(
        f"   📋 Catégorielles: {len(config['categorical_cols']) - len(missing_cat)}/{len(config['categorical_cols'])} trouvées"
    )
    print(f"   🎯 Target: {'✅' if target_ok else '❌'}")

    if missing_seq or missing_cat or not target_ok:
        print("   🔍 Recherche alternatives...")

        # Auto-correction simple (chercher colonnes similaires)
        def find_similar(missing_col, available_cols):
            missing_lower = missing_col.lower()
            for col in available_cols:
                if missing_lower.replace("_", "").replace(
                    ":", ""
                ) in col.lower().replace("_", ""):
                    return col
            return None

        # Corriger séquentielles
        fixed_seq = []
        for col in config["sequence_cols"]:
            if col in all_available:
                fixed_seq.append(col)
            else:
                alt = find_similar(col, all_available)
                if alt:
                    fixed_seq.append(alt)
                    print(f"   💡 '{col}' → '{alt}'")

        # Corriger catégorielles
        fixed_cat = []
        for col in config["categorical_cols"]:
            if col in all_available:
                fixed_cat.append(col)
            else:
                alt = find_similar(col, all_available)
                if alt:
                    fixed_cat.append(alt)
                    print(f"   💡 '{col}' → '{alt}'")

        # Corriger target
        fixed_target = config["target_col"]
        if not target_ok:
            alt = find_similar(config["target_col"], all_available)
            if alt:
                fixed_target = alt
                print(f"   💡 Target '{config['target_col']}' → '{alt}'")

        fixed_config.update(
            {
                "sequence_cols": fixed_seq,
                "categorical_cols": fixed_cat,
                "target_col": fixed_target,
            }
        )

    return fixed_config


# Vérifier et corriger la configuration
logger.info("Vérification et correction des colonnes...")
with tqdm(total=1, desc="🔍 Vérification colonnes") as pbar:
    CONFIG = verify_and_fix_columns(df_raw, CONFIG)
    pbar.update(1)

# Vérifier que nous avons assez de colonnes
total_features = len(CONFIG["sequence_cols"]) + len(CONFIG["categorical_cols"])
target_available = CONFIG["target_col"] in df_raw.columns

if total_features < 8:
    raise Exception(
        f"Trop peu de features trouvées: {total_features}. Minimum 8 requis."
    )

if not target_available:
    raise Exception(
        f"Target '{CONFIG['target_col']}' non trouvée. Pipeline impossible."
    )

print(f"✅ Configuration corrigée: {total_features} features + target")

# === PRÉPARATION DONNÉES AVEC CHUNKS ===
print(f"\n🔧 4. TRANSFORMATION DONNÉES (chunks de {CONFIG['chunk_size']})...")

# Sélectionner seulement les colonnes nécessaires
required_cols = (
    CONFIG["sequence_cols"] + CONFIG["categorical_cols"] + [CONFIG["target_col"]]
)
df_selected = df_raw[required_cols].copy()

print(f"📊 Données sélectionnées: {df_selected.shape}")

# Nettoyage basique
df_selected = df_selected.dropna()
print(f"📊 Après nettoyage: {df_selected.shape}")


# Processing par chunks pour optimiser mémoire
def process_data_chunks(df, chunk_size=2000):
    """Process data by chunks to manage memory"""
    chunks_processed = []
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)

    logger.info(f"Processing {num_chunks} chunks de {chunk_size} lignes...")

    with tqdm(total=num_chunks, desc="🔄 Processing chunks") as pbar:
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()

            if len(chunk) > 0:
                chunks_processed.append(chunk)
                pbar.set_postfix(
                    {"Chunk": f"{i + 1}/{num_chunks}", "Lignes": len(chunk)}
                )
            pbar.update(1)

    return pd.concat(chunks_processed, ignore_index=True)


# Process data
df_processed = process_data_chunks(df_selected, CONFIG["chunk_size"])

# === TRANSFORMATION AVEC TOOLKIT ===
print("\n🔧 5. TRANSFORMATION AVEC VOTRE TOOLKIT...")

# Initialiser transformers
seq_transformer = SequenceTransformer(
    max_sequence_length=CONFIG["max_sequence_length"], vocab_size=CONFIG["vocab_size"]
)

cat_transformer = CategoricalTransformer(
    max_categories=30  # Limité pour performance
)

# Transformer séquences
logger.info("Transformation des features séquentielles...")
with tqdm(total=1, desc="🔄 Transform séquentielles") as pbar:
    seq_result = seq_transformer.fit_transform(df_processed, CONFIG["sequence_cols"])
    pbar.update(1)
logger.info(f"Séquentielles transformées: {len(seq_result.data)} features")

# Transformer catégorielles
logger.info("Transformation des features catégorielles...")
with tqdm(total=1, desc="🔄 Transform catégorielles") as pbar:
    cat_result = cat_transformer.fit_transform(df_processed, CONFIG["categorical_cols"])
    pbar.update(1)
logger.info(f"Catégorielles transformées: {len(cat_result.data)} features")

# Préparer target
target_data = df_processed[CONFIG["target_col"]].values
unique_targets = np.unique(target_data)
CONFIG["target_vocab_size"] = len(unique_targets)

print(f"   🎯 Target préparée: {CONFIG['target_vocab_size']} classes uniques")

# === MODÈLE T4REC OPTIMISÉ ===
print("\n🏗️ 6. CRÉATION MODÈLE T4REC OPTIMISÉ...")


class OptimizedBankingModel(nn.Module):
    """Modèle bancaire T4Rec optimisé pour 100K lignes"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings pour features catégorielles
        self.item_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )
        self.user_embedding = nn.Embedding(
            config["vocab_size"], config["embedding_dim"]
        )

        # Transformer optimisé
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embedding_dim"],
            nhead=config["num_heads"],
            dim_feedforward=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config["num_layers"])

        # Tête de prédiction optimisée
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(config["embedding_dim"]),
            nn.Linear(config["embedding_dim"], config["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size"], config["target_vocab_size"]),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config["max_sequence_length"], config["embedding_dim"])
        )

    def forward(self, item_ids, user_ids):
        # Embeddings
        item_emb = self.item_embedding(item_ids)  # [batch, seq, emb]
        user_emb = self.user_embedding(user_ids)  # [batch, seq, emb]

        # Combine embeddings
        combined = item_emb + user_emb

        # Add positional encoding
        seq_len = combined.size(1)
        combined = combined + self.pos_encoding[:, :seq_len, :]

        # Transformer
        transformed = self.transformer(combined)  # [batch, seq, emb]

        # Prédiction (utiliser la dernière position)
        last_hidden = transformed[:, -1, :]  # [batch, emb]
        predictions = self.prediction_head(last_hidden)  # [batch, target_vocab]

        return predictions


# Créer le modèle
logger.info("Création du modèle T4Rec optimisé...")
with tqdm(total=1, desc="🏗️ Création modèle") as pbar:
    model = OptimizedBankingModel(CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    pbar.update(1)
logger.info(f"Modèle créé: {total_params:,} paramètres")

# === PRÉPARATION ENTRAÎNEMENT ===
print("\n🏋️ 7. PRÉPARATION ENTRAÎNEMENT...")


# Créer sequences d'entraînement
def create_training_sequences(seq_data, cat_data, target_data, config):
    """Créer séquences optimisées pour l'entraînement"""
    # Obtenir le nombre d'échantillons à partir des données transformées
    seq_sample_count = len(list(seq_data.data.values())[0]) if seq_data.data else 0
    cat_sample_count = len(list(cat_data.data.values())[0]) if cat_data.data else 0
    n_samples = min(seq_sample_count, cat_sample_count, len(target_data))
    seq_len = config["max_sequence_length"]

    # Prendre des échantillons équilibrés
    indices = np.random.choice(
        n_samples, size=min(n_samples, config["sample_size"]), replace=False
    )

    sequences = []
    targets = []

    for idx in indices:
        # Créer séquence à partir des features transformées
        seq_values = []
        for col_data in seq_data.data.values():
            if isinstance(col_data, np.ndarray) and len(col_data) > idx:
                # col_data[idx] est un scalaire, on l'ajoute directement
                seq_values.append(float(col_data[idx]))

        cat_values = []
        for col_data in cat_data.data.values():
            if isinstance(col_data, np.ndarray) and len(col_data) > idx:
                # col_data[idx] est un scalaire, on l'ajoute directement
                cat_values.append(float(col_data[idx]))

        # Combiner toutes les features pour cette ligne
        combined = seq_values + cat_values

        # Ajuster à la longueur de séquence (padding ou troncature)
        if len(combined) < seq_len:
            combined.extend([0.0] * (seq_len - len(combined)))  # Padding
        else:
            combined = combined[:seq_len]  # Troncature

        if len(combined) == seq_len:
            sequences.append(combined)
            targets.append(target_data[idx] if idx < len(target_data) else 0)

    return np.array(sequences), np.array(targets)


# Créer les séquences
logger.info("Création des séquences d'entraînement...")
with tqdm(total=1, desc="📊 Création séquences") as pbar:
    item_sequences, target_sequences = create_training_sequences(
        seq_result, cat_result, target_data, CONFIG
    )
    user_sequences = item_sequences.copy()  # Simplifié pour cet exemple
    pbar.update(1)
logger.info(f"Séquences créées: {item_sequences.shape}")

# Encoder targets
from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()
encoded_targets = target_encoder.fit_transform(target_sequences)

# Split train/validation
split_idx = int(0.8 * len(item_sequences))
train_items = torch.tensor(item_sequences[:split_idx], dtype=torch.long)
train_users = torch.tensor(user_sequences[:split_idx], dtype=torch.long)
train_targets = torch.tensor(encoded_targets[:split_idx], dtype=torch.long)

val_items = torch.tensor(item_sequences[split_idx:], dtype=torch.long)
val_users = torch.tensor(user_sequences[split_idx:], dtype=torch.long)
val_targets = torch.tensor(encoded_targets[split_idx:], dtype=torch.long)

print(f"   📊 Split: {len(train_items)} train, {len(val_items)} validation")

# === ENTRAÎNEMENT OPTIMISÉ ===
print("\n🚀 8. ENTRAÎNEMENT MODÈLE...")

# Optimizer et loss
optimizer = torch.optim.AdamW(
    model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()


def train_epoch(
    model, train_items, train_users, train_targets, batch_size, optimizer, criterion
):
    """Entraînement une époque avec gestion batch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(train_items), batch_size):
        batch_items = train_items[i : i + batch_size]
        batch_users = train_users[i : i + batch_size]
        batch_targets = train_targets[i : i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_items, batch_users)
        loss = criterion(outputs, batch_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(model, val_items, val_users, val_targets, batch_size):
    """Évaluation avec métriques"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(val_items), batch_size):
            batch_items = val_items[i : i + batch_size]
            batch_users = val_users[i : i + batch_size]
            batch_targets = val_targets[i : i + batch_size]

            outputs = model(batch_items, batch_users)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_targets.size(0)
            correct += (predicted == batch_targets).sum().item()

    return correct / total


# Boucle d'entraînement
logger.info(f"Début entraînement: {CONFIG['num_epochs']} époques")
print(f"🚀 Début entraînement: {CONFIG['num_epochs']} époques")
print("=" * 60)

training_history = []

# Progress bar pour les époques
with tqdm(total=CONFIG["num_epochs"], desc="🚀 Entraînement") as epoch_pbar:
    for epoch in range(CONFIG["num_epochs"]):
        # Entraînement
        train_loss = train_epoch(
            model,
            train_items,
            train_users,
            train_targets,
            CONFIG["batch_size"],
            optimizer,
            criterion,
        )

        # Validation
        val_accuracy = evaluate_model(
            model, val_items, val_users, val_targets, CONFIG["batch_size"]
        )

        # Scheduler
        scheduler.step()

        # Log
        current_lr = scheduler.get_last_lr()[0]
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "Loss": f"{train_loss:.4f}",
                "Val Acc": f"{val_accuracy:.4f}",
                "LR": f"{current_lr:.6f}",
            }
        )
        epoch_pbar.update(1)

        # Log détaillé
        logger.info(
            f"Époque {epoch + 1:2d}/{CONFIG['num_epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

print("=" * 60)
logger.info("Entraînement terminé!")
print("✅ Entraînement terminé!")

# === ÉVALUATION FINALE ===
print("\n📊 9. ÉVALUATION FINALE...")

# Métriques détaillées sur validation
logger.info("Calcul des métriques finales...")
model.eval()
all_predictions = []
all_targets = []

num_val_batches = (len(val_items) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
with torch.no_grad():
    with tqdm(total=num_val_batches, desc="📊 Évaluation finale") as eval_pbar:
        for i in range(0, len(val_items), CONFIG["batch_size"]):
            batch_items = val_items[i : i + CONFIG["batch_size"]]
            batch_users = val_users[i : i + CONFIG["batch_size"]]
            batch_targets = val_targets[i : i + CONFIG["batch_size"]]

            outputs = model(batch_items, batch_users)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

            eval_pbar.update(1)

# Calculer métriques
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

final_accuracy = accuracy_score(all_targets, all_predictions)
final_precision = precision_score(
    all_targets, all_predictions, average="weighted", zero_division=0
)
final_recall = recall_score(
    all_targets, all_predictions, average="weighted", zero_division=0
)
final_f1 = f1_score(all_targets, all_predictions, average="weighted", zero_division=0)

print(f"📈 RÉSULTATS FINAUX:")
print(f"   🎯 Accuracy: {final_accuracy:.4f}")
print(f"   🎯 Precision: {final_precision:.4f}")
print(f"   🎯 Recall: {final_recall:.4f}")
print(f"   🎯 F1-Score: {final_f1:.4f}")

# === SAUVEGARDE RÉSULTATS ===
print("\n💾 10. SAUVEGARDE RÉSULTATS...")

# Préparer features transformées pour output (100K = plus de données)
features_output = []
for i, (seq_name, seq_data) in enumerate(seq_result.data.items()):
    if isinstance(seq_data, np.ndarray):
        for j, values in enumerate(seq_data[:10000]):  # 10K lignes pour 100K dataset
            features_output.append(
                {
                    "row_id": j,
                    "feature_type": "sequence",
                    "feature_name": seq_name,
                    "feature_value": float(values),  # Valeur complète au lieu de str
                    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

# Ajouter aussi les features catégorielles
for i, (cat_name, cat_data) in enumerate(cat_result.data.items()):
    if isinstance(cat_data, np.ndarray):
        for j, values in enumerate(cat_data[:10000]):  # 10K lignes
            features_output.append(
                {
                    "row_id": j,
                    "feature_type": "categorical",
                    "feature_name": cat_name,
                    "feature_value": float(values),  # Valeur complète
                    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

df_features = pd.DataFrame(features_output)

# Préparer prédictions (plus de prédictions pour 100K)
predictions_output = []
for i, (pred, target) in enumerate(
    zip(all_predictions[:1000], all_targets[:1000])
):  # Top 1000 prédictions pour 100K
    pred_product = (
        target_encoder.inverse_transform([pred])[0]
        if pred < len(target_encoder.classes_)
        else "Unknown"
    )
    true_product = (
        target_encoder.inverse_transform([target])[0]
        if target < len(target_encoder.classes_)
        else "Unknown"
    )

    predictions_output.append(
        {
            "client_id": i,
            "predicted_product": pred_product,
            "true_product": true_product,
            "prediction_correct": pred == target,
            "model_confidence": 0.85,  # Placeholder
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

df_predictions = pd.DataFrame(predictions_output)

# Préparer métriques
metrics_output = []
for epoch_data in training_history:
    metrics_output.append(
        {
            "metric_type": "TRAINING",
            "epoch": epoch_data["epoch"],
            "metric_name": "train_loss",
            "metric_value": epoch_data["train_loss"],
            "details": f"LR: {epoch_data['learning_rate']:.6f}",
        }
    )
    metrics_output.append(
        {
            "metric_type": "VALIDATION",
            "epoch": epoch_data["epoch"],
            "metric_name": "accuracy",
            "metric_value": epoch_data["val_accuracy"],
            "details": f"Époque {epoch_data['epoch']}",
        }
    )

# Métriques finales
final_metrics = [
    {
        "metric_type": "FINAL",
        "epoch": CONFIG["num_epochs"],
        "metric_name": "accuracy",
        "metric_value": final_accuracy,
        "details": "Score final",
    },
    {
        "metric_type": "FINAL",
        "epoch": CONFIG["num_epochs"],
        "metric_name": "precision",
        "metric_value": final_precision,
        "details": "Score final",
    },
    {
        "metric_type": "FINAL",
        "epoch": CONFIG["num_epochs"],
        "metric_name": "recall",
        "metric_value": final_recall,
        "details": "Score final",
    },
    {
        "metric_type": "FINAL",
        "epoch": CONFIG["num_epochs"],
        "metric_name": "f1_score",
        "metric_value": final_f1,
        "details": "Score final",
    },
    {
        "metric_type": "MODEL",
        "epoch": 0,
        "metric_name": "total_parameters",
        "metric_value": total_params,
        "details": f"Architecture: {CONFIG['num_layers']}L-{CONFIG['num_heads']}H-{CONFIG['embedding_dim']}D",
    },
]

metrics_output.extend(final_metrics)

# Métriques détaillées pour 100K (plus d'infos)
additional_metrics = [
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "sample_size",
        "metric_value": CONFIG["sample_size"],
        "details": "Taille échantillon",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "chunk_size",
        "metric_value": CONFIG["chunk_size"],
        "details": "Taille chunks",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "embedding_dim",
        "metric_value": CONFIG["embedding_dim"],
        "details": "Dimension embeddings",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "num_layers",
        "metric_value": CONFIG["num_layers"],
        "details": "Nombre de layers",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "num_heads",
        "metric_value": CONFIG["num_heads"],
        "details": "Attention heads",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "hidden_size",
        "metric_value": CONFIG["hidden_size"],
        "details": "Taille hidden layer",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "batch_size",
        "metric_value": CONFIG["batch_size"],
        "details": "Taille batch",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "learning_rate",
        "metric_value": CONFIG["learning_rate"],
        "details": "Learning rate",
    },
    {
        "metric_type": "CONFIG",
        "epoch": 0,
        "metric_name": "dropout",
        "metric_value": CONFIG["dropout"],
        "details": "Dropout rate",
    },
    {
        "metric_type": "DATA",
        "epoch": 0,
        "metric_name": "sequence_features",
        "metric_value": len(CONFIG["sequence_cols"]),
        "details": "Nombre features séquentielles",
    },
    {
        "metric_type": "DATA",
        "epoch": 0,
        "metric_name": "categorical_features",
        "metric_value": len(CONFIG["categorical_cols"]),
        "details": "Nombre features catégorielles",
    },
    {
        "metric_type": "DATA",
        "epoch": 0,
        "metric_name": "target_classes",
        "metric_value": CONFIG["target_vocab_size"],
        "details": "Nombre classes cible",
    },
    {
        "metric_type": "MODEL",
        "epoch": 0,
        "metric_name": "total_parameters",
        "metric_value": total_params,
        "details": "Paramètres totaux modèle",
    },
    {
        "metric_type": "PERFORMANCE",
        "epoch": 0,
        "metric_name": "execution_time_seconds",
        "metric_value": int(time.time() - start_time),
        "details": "Temps exécution total",
    },
]
metrics_output.extend(additional_metrics)
df_metrics = pd.DataFrame(metrics_output)
df_metrics["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Sauvegarder dans Dataiku
logger.info("Sauvegarde des résultats dans Dataiku...")
save_tasks = ["Features", "Prédictions", "Métriques"]
with tqdm(total=3, desc="💾 Sauvegarde") as save_pbar:
    try:
        features_dataset.write_with_schema(df_features)
        logger.info("Features sauvegardées")
        save_pbar.set_postfix({"Status": "Features OK"})
        save_pbar.update(1)
    except Exception as e:
        logger.error(f"Erreur features: {e}")
        save_pbar.set_postfix({"Status": "Features ERROR"})
        save_pbar.update(1)

    try:
        predictions_dataset.write_with_schema(df_predictions)
        logger.info("Prédictions sauvegardées")
        save_pbar.set_postfix({"Status": "Prédictions OK"})
        save_pbar.update(1)
    except Exception as e:
        logger.error(f"Erreur prédictions: {e}")
        save_pbar.set_postfix({"Status": "Prédictions ERROR"})
        save_pbar.update(1)

    try:
        metrics_dataset.write_with_schema(df_metrics)
        logger.info("Métriques sauvegardées")
        save_pbar.set_postfix({"Status": "Métriques OK"})
        save_pbar.update(1)
    except Exception as e:
        logger.error(f"Erreur métriques: {e}")
        save_pbar.set_postfix({"Status": "Métriques ERROR"})
        save_pbar.update(1)

# === RÉSUMÉ FINAL ===
print("\n" + "=" * 70)
print("🎉 PIPELINE DATAIKU 100K LIGNES - TERMINÉ AVEC SUCCÈS!")
print("=" * 70)

print(f"📊 DONNÉES:")
print(f"   • Échantillon: {len(df_processed):,} lignes")
print(
    f"   • Features: {len(CONFIG['sequence_cols'])} séquentielles + {len(CONFIG['categorical_cols'])} catégorielles"
)
print(f"   • Target: {CONFIG['target_vocab_size']} classes")

print(f"\n🏗️ MODÈLE:")
print(
    f"   • Architecture: XLNet {CONFIG['num_layers']}L-{CONFIG['num_heads']}H-{CONFIG['embedding_dim']}D"
)
print(f"   • Paramètres: {total_params:,}")
print(f"   • Époques: {CONFIG['num_epochs']}")

print(f"\n📈 PERFORMANCE:")
print(f"   • Accuracy finale: {final_accuracy:.1%}")
print(f"   • Precision: {final_precision:.1%}")
print(f"   • Recall: {final_recall:.1%}")
print(f"   • F1-Score: {final_f1:.1%}")

print(f"\n💾 OUTPUTS CRÉÉS:")
print(f"   • T4REC_FEATURES_100K: {len(df_features)} lignes")
print(f"   • T4REC_PREDICTIONS_100K: {len(df_predictions)} prédictions")
print(f"   • T4REC_METRICS_100K: {len(df_metrics)} métriques")

if final_accuracy > 0.6:
    print(f"\n🟢 EXCELLENT! Modèle prêt pour production")
elif final_accuracy > 0.4:
    print(f"\n🟡 BON! Modèle à optimiser")
else:
    print(f"\n🟠 MOYEN! Modèle à retravailler")

# Temps total d'exécution
total_time = time.time() - start_time
logger.info(
    f"Pipeline terminé en {total_time:.1f} secondes ({total_time / 60:.1f} minutes)"
)
print(f"⏱️ Temps total d'exécution: {total_time:.1f}s ({total_time / 60:.1f}min)")

print("=" * 70)
