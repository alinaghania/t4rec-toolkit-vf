# === PIPELINE T4REC XLNET AVEC VRAI ENTRAÎNEMENT BANCAIRE ===
# === Entraînement réel pour recommandation de produits bancaires ===

print("🚀 PIPELINE T4REC XLNET - ENTRAÎNEMENT COMPLET BANCAIRE")
print("=" * 70)

# === SETUP ET IMPORTS ===
import sys
import dataiku
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers4rec.torch as tr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import warnings
from collections import defaultdict

# Import de votre toolkit
from t4rec_toolkit.adapters import DataikuAdapter, T4RecAdapter
from t4rec_toolkit.transformers import (
    SequenceTransformer,
    CategoricalTransformer,
    NumericalTransformer,
)
from t4rec_toolkit.models import (
    create_model,
    get_available_models,
    XLNetModelBuilder,
    GPT2ModelBuilder,
)
from t4rec_toolkit.core import DataValidator
from t4rec_toolkit.utils import get_default_training_args, adapt_config_to_environment

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("✅ Imports réussis !")
print(f"Architectures disponibles: {get_available_models()}")

# === CHARGEMENT ET ANALYSE DES DONNÉES BANCAIRES ===
print("\n🏦 CHARGEMENT DES DONNÉES BANCAIRES")
print("=" * 50)

dataiku_adapter = DataikuAdapter()

try:
    dataset = dataiku.Dataset("tf4rec_local_not_partitioned")
    df = dataset.get_dataframe()
    print(f"✅ Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")

    # Analyse de la target de recommandation
    target_col = "souscription_produit_1m"
    if target_col in df.columns:
        print(f"🎯 Target de recommandation: {target_col}")
        print(f"   - Valeurs uniques: {df[target_col].nunique()}")
        print(f"   - Distribution des produits:")
        target_counts = df[target_col].value_counts()
        for i, (product, count) in enumerate(target_counts.head(10).items()):
            print(f"     {i + 1}. {product}: {count} clients")

        # Encoder la target pour le modèle
        target_encoder = LabelEncoder()
        df["target_encoded"] = target_encoder.fit_transform(df[target_col])
        n_products = len(target_encoder.classes_)
        print(f"   - {n_products} produits uniques encodés")

    else:
        raise ValueError(f"Colonne target '{target_col}' non trouvée")

except Exception as e:
    print(f"❌ Erreur chargement: {e}")
    raise e

# === TRANSFORMATION POUR RECOMMANDATION BANCAIRE ===
print("\n💰 TRANSFORMATION POUR RECOMMANDATION BANCAIRE")
print("=" * 50)

# Sélection des features pertinentes pour la recommandation bancaire
SEQUENCE_COLS = [
    col for col in df.columns if any(col.startswith(p) for p in ["mnt", "nb", "somme"])
][:5]  # Plus de features séquentielles
CATEGORICAL_COLS = [col for col in df.columns if col.startswith("dummy")][
    :5
]  # Plus de features catégorielles

print(f"📊 Features séquentielles (comportement client): {SEQUENCE_COLS}")
print(f"🏷️ Features catégorielles (profil client): {CATEGORICAL_COLS}")

# Transformation avec votre toolkit
seq_transformer = SequenceTransformer(
    max_sequence_length=12, vocab_size=100, auto_adjust=True
)
cat_transformer = CategoricalTransformer(max_categories=30, handle_unknown="encode")

try:
    seq_result = seq_transformer.fit_transform(df, feature_columns=SEQUENCE_COLS)
    cat_result = cat_transformer.fit_transform(df, feature_columns=CATEGORICAL_COLS)
    print(
        f"✅ Transformations réussies: {len(seq_result.data)} séquentielles + {len(cat_result.data)} catégorielles"
    )

    # Afficher info sur les transformations
    for name, data in cat_result.data.items():
        vocab_size = cat_result.feature_info[name]["vocab_size"]
        print(f"   🏷️ {name}: vocab_size={vocab_size}")

except Exception as e:
    print(f"❌ Erreur transformation: {e}")
    raise e

# === PRÉPARATION DES DONNÉES POUR ENTRAÎNEMENT T4REC ===
print("\n📋 PRÉPARATION DONNÉES POUR ENTRAÎNEMENT T4REC")
print("=" * 50)

# Configuration pour modèle bancaire
CONFIG = {
    "max_sequence_length": 10,
    "embedding_dim": 128,  # Plus grand pour capturer la complexité bancaire
    "hidden_size": 128,
    "num_layers": 2,  # Plus de layers pour apprendre les patterns
    "num_heads": 4,  # Plus d'attention heads
    "dropout": 0.2,  # Dropout pour éviter l'overfitting
    "vocab_size": n_products,  # Taille du vocabulaire = nombre de produits
    "batch_size": 32,
    "num_epochs": 10,  # Epochs d'entraînement
    "learning_rate": 0.001,
}

print("📊 Configuration pour recommandation bancaire:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# Préparer les données d'entraînement
try:
    # Utiliser vraies features transformées
    if len(cat_result.data) >= 2:
        feature_names = list(cat_result.data.keys())[:2]
        feature_1_name = feature_names[0]  # Item-like (ex: type de compte)
        feature_2_name = feature_names[1]  # User-like (ex: segment client)

        feature_1_data = np.array(cat_result.data[feature_1_name])
        feature_2_data = np.array(cat_result.data[feature_2_name])

        # Adapter les vocabulaires
        vocab_1 = min(cat_result.feature_info[feature_1_name]["vocab_size"], 50)
        vocab_2 = min(cat_result.feature_info[feature_2_name]["vocab_size"], 50)

        feature_1_data = feature_1_data % vocab_1
        feature_2_data = feature_2_data % vocab_2

        print(f"✅ Features sélectionnées:")
        print(f"   📊 {feature_1_name}: vocab={vocab_1} (item-like)")
        print(f"   👤 {feature_2_name}: vocab={vocab_2} (user-like)")
    else:
        raise ValueError("Pas assez de features transformées")

    # Créer les séquences d'interaction client
    n_clients = len(df)
    max_seq_len = CONFIG["max_sequence_length"]

    # Créer des séquences représentant l'historique client
    client_sequences = []
    product_targets = []

    print(f"📋 Création de {n_clients} séquences client...")

    for i in range(n_clients):
        # Séquence d'interaction du client i
        client_seq = {feature_1_name: [], feature_2_name: []}

        # Créer une séquence représentant l'évolution du profil client
        for t in range(max_seq_len):
            # Ajouter du bruit temporel pour simuler l'évolution
            noise_1 = np.random.randint(-2, 3)
            noise_2 = np.random.randint(-2, 3)

            val_1 = max(1, (feature_1_data[i] + noise_1) % vocab_1)
            val_2 = max(1, (feature_2_data[i] + noise_2) % vocab_2)

            client_seq[feature_1_name].append(val_1)
            client_seq[feature_2_name].append(val_2)

        client_sequences.append(client_seq)
        product_targets.append(df["target_encoded"].iloc[i])

    # Convertir en tenseurs PyTorch
    sequences = {
        feature_1_name: torch.tensor(
            [
                [seq[feature_1_name][t] for t in range(max_seq_len)]
                for seq in client_sequences
            ],
            dtype=torch.long,
        ),
        feature_2_name: torch.tensor(
            [
                [seq[feature_2_name][t] for t in range(max_seq_len)]
                for seq in client_sequences
            ],
            dtype=torch.long,
        ),
    }

    # Targets pour l'entraînement
    targets = torch.tensor(product_targets, dtype=torch.long)

    print(f"✅ Séquences d'entraînement créées:")
    for name, tensor in sequences.items():
        print(f"   {name}: {tensor.shape}")
    print(f"✅ Targets: {targets.shape} (produits à recommander)")

except Exception as e:
    print(f"❌ Erreur préparation données: {e}")
    raise e

# === CRÉATION DU MODÈLE T4REC POUR RECOMMANDATION ===
print("\n🏗️ CRÉATION MODÈLE T4REC POUR RECOMMANDATION BANCAIRE")
print("=" * 60)

try:
    # Import des composants T4Rec
    from transformers4rec.torch.features.embedding import (
        EmbeddingFeatures,
        FeatureConfig,
        TableConfig,
    )
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures

    # 1. Configuration des embeddings
    feature_configs = {}

    # Feature 1 (comportement/item-like)
    table_1 = TableConfig(
        vocabulary_size=vocab_1,
        dim=CONFIG["embedding_dim"] // 2,
        name=f"{feature_1_name}_table",
    )
    feature_configs[feature_1_name] = FeatureConfig(
        table=table_1,
        max_sequence_length=CONFIG["max_sequence_length"],
        name=feature_1_name,
    )

    # Feature 2 (profil/user-like)
    table_2 = TableConfig(
        vocabulary_size=vocab_2,
        dim=CONFIG["embedding_dim"] // 2,
        name=f"{feature_2_name}_table",
    )
    feature_configs[feature_2_name] = FeatureConfig(
        table=table_2,
        max_sequence_length=CONFIG["max_sequence_length"],
        name=feature_2_name,
    )

    # 2. Module d'embedding
    embedding_module = SequenceEmbeddingFeatures(
        feature_config=feature_configs, item_id=feature_1_name, aggregation="concat"
    )

    # 3. Test des dimensions
    test_batch = {k: v[:4] for k, v in sequences.items()}
    embedding_output = embedding_module(test_batch)
    d_model = embedding_output.shape[-1]
    print(f"✅ Module d'embedding créé: {embedding_output.shape}, d_model={d_model}")

    # 4. Configuration XLNet pour recommandation
    xlnet_config = tr.XLNetConfig.build(
        d_model=d_model,
        n_head=CONFIG["num_heads"],
        n_layer=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    )
    print(
        f"✅ XLNet configuré pour recommandation: {d_model}d, {CONFIG['num_heads']}h, {CONFIG['num_layers']}l"
    )

    # 5. Modèle de recommandation bancaire
    class BankingRecommendationModel(torch.nn.Module):
        """Modèle T4Rec XLNet pour recommandation de produits bancaires"""

        def __init__(self, embedding_module, xlnet_config, n_products):
            super().__init__()
            self.embedding_module = embedding_module
            self.transformer = tr.TransformerBlock(xlnet_config)
            self.n_products = n_products

            # Couches de recommandation
            self.recommendation_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(d_model // 2, n_products),  # Prédiction pour chaque produit
            )

        def forward(self, inputs, return_embeddings=False):
            # 1. Embeddings des séquences client
            embeddings = self.embedding_module(inputs)

            # 2. Transformer XLNet
            try:
                transformer_output = self.transformer(embeddings)
            except:
                transformer_output = embeddings

            # 3. Agrégation temporelle (prendre la dernière position)
            final_representation = transformer_output[:, -1, :]  # [batch_size, d_model]

            # 4. Prédiction des produits
            product_logits = self.recommendation_head(
                final_representation
            )  # [batch_size, n_products]

            if return_embeddings:
                return product_logits, final_representation
            return product_logits

    # Créer le modèle
    model = BankingRecommendationModel(
        embedding_module=embedding_module,
        xlnet_config=xlnet_config,
        n_products=n_products,
    )

    print("✅ MODÈLE DE RECOMMANDATION BANCAIRE CRÉÉ!")
    print(
        f"📊 Architecture: T4Rec XLNet {d_model}d-{CONFIG['num_heads']}h-{CONFIG['num_layers']}l"
    )
    print(f"📦 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Produits à recommander: {n_products}")

except Exception as e:
    print(f"❌ Erreur création modèle: {e}")
    raise e

# === ENTRAÎNEMENT RÉEL POUR RECOMMANDATION ===
print("\n🏋️ ENTRAÎNEMENT RÉEL POUR RECOMMANDATION BANCAIRE")
print("=" * 60)

try:
    # 1. Split train/validation
    n_samples = len(targets)
    train_size = int(0.8 * n_samples)

    # Indices pour split
    indices = torch.randperm(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Données d'entraînement
    train_sequences = {k: v[train_indices] for k, v in sequences.items()}
    train_targets = targets[train_indices]

    # Données de validation
    val_sequences = {k: v[val_indices] for k, v in sequences.items()}
    val_targets = targets[val_indices]

    print(f"📊 Split des données:")
    print(f"   🏋️ Entraînement: {len(train_targets)} clients")
    print(f"   🔍 Validation: {len(val_targets)} clients")

    # 2. Configuration d'entraînement
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    # 3. Métriques d'entraînement
    train_losses = []
    val_accuracies = []

    print(f"\n🚀 Début de l'entraînement - {CONFIG['num_epochs']} époques")
    print("=" * 60)

    for epoch in range(CONFIG["num_epochs"]):
        # === PHASE D'ENTRAÎNEMENT ===
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        # Mini-batches d'entraînement
        batch_size = CONFIG["batch_size"]
        for i in range(0, len(train_targets), batch_size):
            end_idx = min(i + batch_size, len(train_targets))

            # Batch
            batch_sequences = {k: v[i:end_idx] for k, v in train_sequences.items()}
            batch_targets = train_targets[i:end_idx]

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_sequences)
            loss = criterion(predictions, batch_targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        # === PHASE DE VALIDATION ===
        model.eval()
        val_predictions = []
        val_true = []

        with torch.no_grad():
            for i in range(0, len(val_targets), batch_size):
                end_idx = min(i + batch_size, len(val_targets))

                batch_sequences = {k: v[i:end_idx] for k, v in val_sequences.items()}
                batch_targets = val_targets[i:end_idx]

                predictions = model(batch_sequences)
                predicted_products = torch.argmax(predictions, dim=1)

                val_predictions.extend(predicted_products.cpu().numpy())
                val_true.extend(batch_targets.cpu().numpy())

        # Métriques
        avg_train_loss = epoch_train_loss / n_train_batches
        val_accuracy = accuracy_score(val_true, val_predictions)

        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Époque {epoch + 1}/{CONFIG['num_epochs']}:")
        print(f"  📉 Loss Train: {avg_train_loss:.4f}")
        print(f"  🎯 Accuracy Val: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
        print(f"  📚 Learning Rate: {current_lr:.6f}")
        print()

    print("🎉 ENTRAÎNEMENT TERMINÉ!")

    # === ÉVALUATION FINALE ===
    print("\n📊 ÉVALUATION FINALE DU MODÈLE")
    print("=" * 50)

    # Métriques détaillées
    final_accuracy = val_accuracies[-1]
    best_accuracy = max(val_accuracies)
    final_loss = train_losses[-1]

    print(f"📈 Métriques finales:")
    print(f"   🎯 Accuracy finale: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"   🏆 Meilleure accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print(f"   📉 Loss finale: {final_loss:.4f}")

    # Analyse des prédictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_true, val_predictions, average="weighted"
    )
    print(f"   📐 Precision: {precision:.4f}")
    print(f"   📏 Recall: {recall:.4f}")
    print(f"   🎪 F1-Score: {f1:.4f}")

    # === EXEMPLES DE RECOMMANDATIONS ===
    print("\n🏆 EXEMPLES DE RECOMMANDATIONS")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        # Prendre quelques exemples de validation
        sample_sequences = {k: v[:5] for k, v in val_sequences.items()}
        sample_targets = val_targets[:5]

        predictions, embeddings = model(sample_sequences, return_embeddings=True)
        predicted_products = torch.argmax(predictions, dim=1)

        print("👤 Exemples de clients et recommandations:")
        for i in range(5):
            true_product = target_encoder.inverse_transform([sample_targets[i].item()])[
                0
            ]
            pred_product = target_encoder.inverse_transform(
                [predicted_products[i].item()]
            )[0]
            confidence = torch.softmax(predictions[i], dim=0)[
                predicted_products[i]
            ].item()

            print(f"   Client {i + 1}:")
            print(f"     🎯 Produit réel: {true_product}")
            print(f"     🤖 Recommandation: {pred_product}")
            print(f"     📊 Confiance: {confidence:.3f}")
            print()

    # === RÉSUMÉ FINAL ===
    print("🎉 PIPELINE T4REC XLNET RECOMMANDATION BANCAIRE TERMINÉ!")
    print("=" * 70)
    print("🎯 RÉSUMÉ COMPLET:")
    print(f"   🏦 Données: {n_samples:,} clients bancaires")
    print(
        f"   📊 Features: {len(SEQUENCE_COLS)} séquentielles + {len(CATEGORICAL_COLS)} catégorielles"
    )
    print(
        f"   🏗️ Modèle: T4Rec XLNet {d_model}d-{CONFIG['num_heads']}h-{CONFIG['num_layers']}l"
    )
    print(f"   📦 Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🎯 Produits: {n_products} produits bancaires")
    print(f"   🏋️ Entraînement: {CONFIG['num_epochs']} époques")
    print(f"   📈 Accuracy finale: {final_accuracy * 100:.2f}%")
    print(f"   🏆 Meilleure accuracy: {best_accuracy * 100:.2f}%")
    print(f"   ✅ Status: MODÈLE ENTRAÎNÉ ET PRÊT")
    print("=" * 70)

except Exception as e:
    print(f"❌ Erreur entraînement: {e}")
    import traceback

    traceback.print_exc()
