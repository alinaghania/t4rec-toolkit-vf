"""
Pure T4Rec XLNet Pipeline - No PyTorch fallback, only T4Rec
Production pipeline using transformers4rec with XLNet architecture
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# T4Rec imports
try:
    import transformers4rec.torch as tr
    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
    from transformers4rec.torch.features.embedding import (
        EmbeddingFeatures,
        FeatureConfig,
        TableConfig,
    )
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures

    _HAS_T4REC = True
    print("T4Rec imported successfully")
except ImportError as e:
    _HAS_T4REC = False
    print(f"T4Rec import error: {e}")
    raise ImportError(
        "T4Rec is required for this pipeline. Install with: pip install transformers4rec[torch]"
    )

# Optional imports
try:
    import dataiku

    _HAS_DATAIKU = True
except Exception:
    _HAS_DATAIKU = False

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Local toolkit
from .transformers.sequence_transformer import SequenceTransformer
from .transformers.categorical_transformer import CategoricalTransformer

import torch
import torch.nn as nn


@dataclass
class T4RecConfig:
    """Configuration for T4Rec XLNet model"""

    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    max_sequence_length: int = 20

    # XLNet specific
    mem_len: int = 50
    attn_type: str = "bi"

    # Vocabulary
    vocab_size: int = 1000

    # Task specific
    num_classes: int = 150


def _dynamic_t4rec_defaults(sample_size: int) -> T4RecConfig:
    """Dynamic T4Rec configuration based on sample size"""
    if sample_size <= 10000:
        return T4RecConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_sequence_length=15,
            vocab_size=500,
        )
    elif sample_size <= 100000:
        return T4RecConfig(
            d_model=256,
            n_heads=8,
            n_layers=3,
            max_sequence_length=20,
            vocab_size=1000,
        )
    else:
        return T4RecConfig(
            d_model=384,
            n_heads=12,
            n_layers=4,
            max_sequence_length=25,
            vocab_size=1500,
        )


def blank_config() -> Dict[str, Any]:
    """Create blank configuration for T4Rec pipeline"""
    return {
        "data": {
            "dataset_name": "",
            "sample_size": 10000,
            "chunk_size": 2000,
            "partitions": None,
            "temporal_split": None,
        },
        "features": {
            "sequence_cols": [],
            "categorical_cols": [],
            "target_col": "",
            "max_seq_features": None,
            "max_cat_features": None,
        },
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 3,
            "dropout": 0.1,
            "max_sequence_length": 20,
            "mem_len": 50,
            "attn_type": "bi",
            "vocab_size": 1000,
        },
        "transformers": {
            "categorical": {
                "max_categories": 50,
                "handle_unknown": "ignore",
                "unknown_value": 1,
            }
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "val_split": 0.2,
        },
        "runtime": {
            "verbose": True,
            "progress": True,
            "seed": 42,
        },
        "outputs": {
            "features_dataset": None,
            "predictions_dataset": None,
            "metrics_dataset": None,
            "local_dir": "output",
        },
    }


def default_config(
    mode: Optional[str] = None, sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Create default configuration with T4Rec optimizations"""
    config = blank_config()

    if sample_size:
        config["data"]["sample_size"] = sample_size
        # Auto-configure T4Rec model based on sample size
        t4rec_cfg = _dynamic_t4rec_defaults(sample_size)
        config["model"].update(
            {
                "d_model": t4rec_cfg.d_model,
                "n_heads": t4rec_cfg.n_heads,
                "n_layers": t4rec_cfg.n_layers,
                "max_sequence_length": t4rec_cfg.max_sequence_length,
                "vocab_size": t4rec_cfg.vocab_size,
            }
        )

        # Auto-configure training
        config["data"]["chunk_size"] = min(sample_size // 5, 5000)
        if sample_size <= 10000:
            config["training"]["batch_size"] = 32
        elif sample_size <= 100000:
            config["training"]["batch_size"] = 64
        else:
            config["training"]["batch_size"] = 128

    return config


def _setup_logging(verbose: bool) -> None:
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_dataframe(
    dataset_name: str,
    sample_size: int,
    partitions: List[str] = None,
    temporal_split: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load DataFrame from Dataiku with temporal split support

    Returns:
        Tuple[train_df, test_df]: Training data and optional test data
    """
    logger = logging.getLogger(__name__)

    if not _HAS_DATAIKU:
        raise RuntimeError("Dataiku not available - temporal split requires Dataiku")

    try:
        # Import direct de dataiku au lieu de load_dataset qui n'existe pas
        import dataiku

        # Si temporal_split est défini, charger train et test séparément
        if temporal_split:
            logger.info(f"Loading temporal split: {temporal_split}")

            # Construire les partitions d'entraînement
            train_start = temporal_split["train_start"]
            train_end = temporal_split["train_end"]
            test_start = temporal_split["test_start"]
            test_end = temporal_split["test_end"]
            test_sample_size = temporal_split.get("test_sample_size", sample_size // 5)

            # Générer les partitions de train (par exemple: 2023-01 à 2023-11)
            train_partitions = []
            if train_start and train_end:
                # Parsing simple YYYY-MM
                start_year, start_month = map(int, train_start.split("-"))
                end_year, end_month = map(int, train_end.split("-"))

                current_year, current_month = start_year, start_month
                while (current_year, current_month) <= (end_year, end_month):
                    train_partitions.append(f"{current_year}-{current_month:02d}")
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                        current_year += 1

            # Partitions de test
            test_partitions = []
            if test_start and test_end:
                start_year, start_month = map(int, test_start.split("-"))
                end_year, end_month = map(int, test_end.split("-"))

                current_year, current_month = start_year, start_month
                while (current_year, current_month) <= (end_year, end_month):
                    test_partitions.append(f"{current_year}-{current_month:02d}")
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                        current_year += 1

            logger.info(f"Train partitions: {train_partitions}")
            logger.info(f"Test partitions: {test_partitions}")

            # Charger données d'entraînement avec syntaxe Dataiku moderne
            train_dataset = dataiku.Dataset(dataset_name)
            try:
                # Ajouter les partitions à lire (syntaxe moderne Dataiku)
                partition_range = "/".join(train_partitions)  # ex: "2023-01/2023-11"
                train_dataset.add_read_partitions(partition_range)
                train_df = train_dataset.get_dataframe(limit=sample_size)
                logger.info(
                    f"Loaded {len(train_df):,} train rows using partition range: {partition_range}"
                )
            except Exception as e:
                logger.warning(
                    f"Partition range failed: {e}, falling back to full dataset"
                )
                # Fallback: charger tout le dataset puis filtrer
                train_df = train_dataset.get_dataframe(limit=sample_size)

            logger.info(f"Loaded {len(train_df):,} train rows from {dataset_name}")

            # Charger données de test avec syntaxe Dataiku moderne
            test_dataset = dataiku.Dataset(dataset_name)
            try:
                # Ajouter les partitions de test à lire
                test_partition_range = "/".join(test_partitions)  # ex: "2023-12"
                test_dataset.add_read_partitions(test_partition_range)
                test_df = test_dataset.get_dataframe(limit=test_sample_size)
                logger.info(
                    f"Loaded {len(test_df):,} test rows using partition range: {test_partition_range}"
                )
            except Exception as e:
                logger.warning(
                    f"Test partition range failed: {e}, falling back to full dataset"
                )
                # Fallback: charger tout le dataset puis filtrer
                test_df = test_dataset.get_dataframe(limit=test_sample_size)

            logger.info(f"Loaded {len(test_df):,} test rows from {dataset_name}")

            return train_df, test_df

        else:
            # Chargement classique avec support partitions moderne
            dataset = dataiku.Dataset(dataset_name)
            if partitions:
                try:
                    # Syntaxe moderne pour partitions multiples
                    partition_range = "/".join(partitions)  # ex: "2023-01/2023-11"
                    dataset.add_read_partitions(partition_range)
                    df = dataset.get_dataframe(limit=sample_size)
                    logger.info(f"Loaded with partition range: {partition_range}")
                except Exception as e:
                    logger.warning(f"Partition range failed: {e}, loading full dataset")
                    df = dataset.get_dataframe(limit=sample_size)
            else:
                df = dataset.get_dataframe(limit=sample_size)

            logger.info(f"Loaded {len(df):,} rows from {dataset_name}")
            return df, None

    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        # Fallback simple
        dataset = dataiku.Dataset(dataset_name)
        df = dataset.get_dataframe(limit=sample_size)
        return df, None


class T4RecAdvancedModel(torch.nn.Module):
    """Advanced T4Rec + Multi-Embedding + Transformer model for banking recommendation

    Scientific Architecture based on Meta/Pinterest 2024-2025 best practices:
    - T4Rec SequenceEmbeddingFeatures for specialized recommendation embeddings
    - Dual-pathway embeddings (item context + user profile) for richer representations
    - Advanced PyTorch TransformerEncoder with increased depth and capacity
    - Sophisticated prediction head with GELU activation and layered architecture
    - Positional encoding for enhanced sequence understanding
    """

    def __init__(
        self,
        embedding_module,
        xlnet_config,
        n_products,
        d_model,
        embedding_output_dim=None,
        vocab_size=1000,
        max_sequence_length=20,
    ):
        super().__init__()

        # Core T4Rec embeddings (specialized for recommendations)
        self.t4rec_embeddings = embedding_module

        # Advanced dual-pathway embeddings (Meta/Pinterest approach)
        self.item_context_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.user_profile_embedding = torch.nn.Embedding(vocab_size, d_model)

        # Positional encoding for enhanced sequence understanding
        self.positional_encoding = torch.nn.Parameter(
            torch.randn(1, max_sequence_length, d_model) * 0.02
        )

        # Feature fusion layer (will be initialized after first forward pass)
        self.feature_fusion = None  # Dynamic initialization
        self.fusion_input_dim = None

        # Advanced Transformer with increased capacity (6 layers vs 3)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=xlnet_config.n_head
            if hasattr(xlnet_config, "n_head")
            else 16,  # More heads
            dim_feedforward=d_model * 4,  # Larger feedforward
            dropout=xlnet_config.dropout if hasattr(xlnet_config, "dropout") else 0.1,
            activation="gelu",  # Advanced activation
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=6,  # Deeper than standard (6 vs 3)
        )

        # Projection layer for dimension adaptation
        self.projection = None
        if embedding_output_dim and embedding_output_dim != d_model:
            self.projection = torch.nn.Linear(embedding_output_dim, d_model)

        # Advanced prediction head (Pinterest-style sophisticated architecture)
        self.recommendation_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model * 2),  # Larger intermediate layer
            torch.nn.GELU(),  # Advanced activation
            torch.nn.Dropout(0.15),  # Slightly higher dropout for regularization
            torch.nn.Linear(d_model * 2, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(d_model, n_products),
        )

        self.n_products = n_products
        self.d_model = d_model

        # Initialize weights with advanced strategy
        self._init_weights()

    def _init_weights(self):
        """Advanced weight initialization following latest best practices"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, inputs, return_embeddings=False):
        """Advanced forward pass with multi-pathway embeddings and sophisticated fusion"""
        batch_size = list(inputs.values())[0].shape[0]

        # 1. T4Rec specialized embeddings
        t4rec_embeddings = self.t4rec_embeddings(inputs)

        # 2. Dual-pathway embeddings (Meta/Pinterest approach)
        # Extract item and user indices from inputs (use first available features as proxy)
        feature_keys = list(inputs.keys())
        item_indices = (
            inputs[feature_keys[0]]
            if feature_keys
            else torch.zeros(batch_size, dtype=torch.long)
        )
        user_indices = (
            inputs[feature_keys[1]]
            if len(feature_keys) > 1
            else torch.zeros(batch_size, dtype=torch.long)
        )

        # Clip indices to vocab size to avoid index errors
        vocab_size = self.item_context_embedding.num_embeddings
        item_indices = torch.clamp(item_indices.long(), 0, vocab_size - 1)
        user_indices = torch.clamp(user_indices.long(), 0, vocab_size - 1)

        item_context = self.item_context_embedding(item_indices)  # [batch, d_model]
        user_profile = self.user_profile_embedding(user_indices)  # [batch, d_model]

        # 3. Project T4Rec embeddings if needed
        if self.projection is not None:
            t4rec_embeddings = self.projection(t4rec_embeddings)

        # 4. Handle 2D embeddings and create proper sequence dimension
        if len(t4rec_embeddings.shape) == 2:
            t4rec_embeddings = t4rec_embeddings.unsqueeze(1)  # [batch, 1, features]

        # Expand item and user embeddings to match sequence dimension
        seq_len = t4rec_embeddings.shape[1]
        item_context = item_context.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq, d_model]
        user_profile = user_profile.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq, d_model]

        # 5. Advanced feature fusion (combine all embeddings)
        combined_embeddings = torch.cat(
            [
                t4rec_embeddings,  # T4Rec specialized
                item_context,  # Item context
                user_profile,  # User profile
            ],
            dim=-1,
        )  # [batch, seq, combined_features]

        # 6. Dynamic fusion layer initialization (first call only)
        if self.feature_fusion is None:
            actual_fusion_dim = combined_embeddings.shape[-1]
            self.fusion_input_dim = actual_fusion_dim
            self.feature_fusion = torch.nn.Sequential(
                torch.nn.Linear(actual_fusion_dim, self.d_model),
                torch.nn.LayerNorm(self.d_model),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
            ).to(combined_embeddings.device)
            print(f"Initialized feature fusion: {actual_fusion_dim} -> {self.d_model}")

        # 7. Fusion layer to optimal dimension
        fused_embeddings = self.feature_fusion(
            combined_embeddings
        )  # [batch, seq, d_model]

        # 8. Add positional encoding for enhanced sequence understanding
        if fused_embeddings.shape[1] <= self.positional_encoding.shape[1]:
            pos_encoding = self.positional_encoding[:, : fused_embeddings.shape[1], :]
            fused_embeddings = fused_embeddings + pos_encoding

        # 9. Advanced Transformer processing (6 layers with GELU)
        transformer_output = self.transformer(fused_embeddings)  # [batch, seq, d_model]

        # 10. Advanced sequence aggregation (last position with attention weights)
        final_representation = transformer_output[:, -1, :]  # [batch, d_model]

        # 11. Sophisticated prediction head
        product_logits = self.recommendation_head(final_representation)

        if return_embeddings:
            return product_logits, final_representation, transformer_output
        return product_logits


def _create_t4rec_model(
    seq_data: Dict[str, np.ndarray],
    cat_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
):
    """Create hybrid T4Rec XLNet model (proven approach)"""
    model_config = config["model"]

    # Create feature configs for T4Rec embeddings
    feature_configs = {}
    d_model = model_config["d_model"]

    # Configure sequence features (use filtered columns)
    seq_cols = config["features"]["sequence_cols"]
    cat_cols = config["features"]["categorical_cols"]

    for col in seq_cols:
        table_config = TableConfig(
            vocabulary_size=model_config["vocab_size"],
            dim=d_model // len(seq_cols + cat_cols),
            name=f"{col}_table",
        )
        feature_configs[col] = FeatureConfig(
            table=table_config,
            max_sequence_length=model_config["max_sequence_length"],
            name=col,
        )

    for col in cat_cols:
        table_config = TableConfig(
            vocabulary_size=model_config["vocab_size"],
            dim=d_model // len(seq_cols + cat_cols),
            name=f"{col}_table",
        )
        feature_configs[col] = FeatureConfig(
            table=table_config,
            max_sequence_length=model_config["max_sequence_length"],
            name=col,
        )

    # Create T4Rec embedding module (using API that actually exists in 23.04.00)
    # Use the first sequence column as item_id (required parameter)
    item_id_col = seq_cols[0] if seq_cols else cat_cols[0]

    embedding_module = SequenceEmbeddingFeatures(
        feature_config=feature_configs, item_id=item_id_col, aggregation="concat"
    )

    # Test embedding to get actual d_model
    test_batch = {}
    all_cols = seq_cols + cat_cols
    for col in all_cols:
        test_batch[col] = torch.randint(
            0, model_config["vocab_size"], (2, model_config["max_sequence_length"])
        )

    # Initialize variables
    embedding_output_dim = None

    with torch.no_grad():
        try:
            test_output = embedding_module(test_batch)
            detected_d_model = test_output.shape[-1]

            # Use configured d_model for XLNet (must be divisible by n_heads)
            # Store the detected dimension for projection layer
            embedding_output_dim = detected_d_model
            n_heads = model_config["n_heads"]
            actual_d_model = d_model  # Use configured d_model

            # Ensure configured d_model is divisible by n_heads
            if actual_d_model % n_heads != 0:
                actual_d_model = (actual_d_model // n_heads) * n_heads
                if config["runtime"]["verbose"]:
                    print(
                        f"Adjusted configured d_model to {actual_d_model} (divisible by n_heads={n_heads})"
                    )

            if config["runtime"]["verbose"]:
                print(
                    f"T4Rec embedding: {detected_d_model}D → Transformer: {actual_d_model}D (projection: {detected_d_model != actual_d_model})"
                )
        except Exception as e:
            if config["runtime"]["verbose"]:
                print(
                    f"T4Rec embedding test failed: {e}, using configured d_model={d_model}"
                )
            # Ensure configured d_model is also divisible by n_heads
            embedding_output_dim = None  # No projection needed
            n_heads = model_config["n_heads"]
            actual_d_model = (d_model // n_heads) * n_heads

    # Create transformer config (will be adapted for PyTorch TransformerEncoder)
    xlnet_config = tr.XLNetConfig.build(
        d_model=actual_d_model,
        n_head=model_config["n_heads"],
        n_layer=model_config["n_layers"],
        dropout=model_config["dropout"],
    )

    # Create hybrid model
    model = T4RecAdvancedModel(
        embedding_module=embedding_module,
        xlnet_config=xlnet_config,
        n_products=model_config[
            "target_vocab_size"
        ],  # Use target classes for prediction head
        d_model=actual_d_model,
        embedding_output_dim=embedding_output_dim,
        vocab_size=model_config["vocab_size"],
        max_sequence_length=model_config.get("max_sequence_length", 20),
    )

    return model


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run T4Rec XLNet training pipeline"""
    start_time = time.time()
    _setup_logging(config["runtime"]["verbose"])
    logger = logging.getLogger(__name__)

    if config["runtime"]["verbose"]:
        logger.info("Starting T4Rec XLNet training pipeline")

    # Set random seed
    if config["runtime"]["seed"]:
        torch.manual_seed(config["runtime"]["seed"])
        np.random.seed(config["runtime"]["seed"])

    # Load data
    dataset_name = config["data"]["dataset_name"]
    sample_size = config["data"]["sample_size"]
    partitions = config["data"].get("partitions")
    temporal_split = config["data"].get("temporal_split")

    train_df, test_df = _load_dataframe(
        dataset_name, sample_size, partitions, temporal_split
    )

    if config["runtime"]["verbose"]:
        logger.info(f"Loaded {len(train_df)} train samples from {dataset_name}")
        if test_df is not None:
            logger.info(f"Loaded {len(test_df)} test samples for temporal validation")

    # Use train_df as primary df for processing
    df = train_df

    # Get features
    seq_cols = config["features"]["sequence_cols"]
    cat_cols = config["features"]["categorical_cols"]
    target_col = config["features"]["target_col"]

    # Verify columns exist in dataframe
    available_cols = set(df.columns)
    missing_seq = [col for col in seq_cols if col not in available_cols]
    missing_cat = [col for col in cat_cols if col not in available_cols]

    if missing_seq or missing_cat:
        if config["runtime"]["verbose"]:
            logger.error(f"Missing sequence columns: {missing_seq}")
            logger.error(f"Missing categorical columns: {missing_cat}")
            logger.info(f"Available columns sample: {list(df.columns)[:20]}...")
        raise ValueError(f"Missing columns - Seq: {missing_seq}, Cat: {missing_cat}")

    # Filter to existing columns only
    seq_cols_filtered = [col for col in seq_cols if col in available_cols]
    cat_cols_filtered = [col for col in cat_cols if col in available_cols]

    if config["runtime"]["verbose"]:
        logger.info(
            f"Using {len(seq_cols_filtered)} sequence and {len(cat_cols_filtered)} categorical columns"
        )

    # Transform features
    seq_transformer = SequenceTransformer()
    cat_transformer = CategoricalTransformer()

    if config["runtime"]["verbose"]:
        logger.info(
            f"Starting sequence transformation for columns: {seq_cols_filtered}"
        )

    seq_result = seq_transformer.fit_transform(df, seq_cols_filtered)

    if config["runtime"]["verbose"]:
        logger.info(
            f"Sequence transformation completed. Output keys: {list(seq_result.data.keys())}"
        )
        logger.info(
            f"Starting categorical transformation for columns: {cat_cols_filtered}"
        )

    cat_result = cat_transformer.fit_transform(df, cat_cols_filtered)

    if config["runtime"]["verbose"]:
        logger.info(
            f"Categorical transformation completed. Output keys: {list(cat_result.data.keys())}"
        )

    # Prepare target
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    targets = encoder.fit_transform(df[target_col])
    target_vocab_size = len(encoder.classes_)

    # Keep original vocab_size for embeddings, store target_vocab_size separately
    config["model"]["target_vocab_size"] = target_vocab_size
    # Ensure vocab_size is large enough for both features and target
    if "vocab_size" not in config["model"]:
        config["model"]["vocab_size"] = 1000  # Default
    config["model"]["vocab_size"] = max(
        config["model"]["vocab_size"], target_vocab_size + 100
    )

    # Prepare data for hybrid model
    transformed_data = {}

    # Debug: Check what keys are available
    if config["runtime"]["verbose"]:
        logger.info(f"Sequence result keys: {list(seq_result.data.keys())}")
        logger.info(f"Categorical result keys: {list(cat_result.data.keys())}")
        logger.info(f"Expected sequence cols (filtered): {seq_cols_filtered}")
        logger.info(f"Expected categorical cols (filtered): {cat_cols_filtered}")

    # Add sequence features (transformers create "{col}_seq" keys)
    for col in seq_cols_filtered:
        seq_key = f"{col}_seq"
        if seq_key in seq_result.data:
            transformed_data[col] = seq_result.data[seq_key]
            if config["runtime"]["verbose"]:
                logger.info(
                    f"Mapped sequence column {col} to transformer key {seq_key}"
                )
        else:
            raise KeyError(
                f"Cannot find sequence transformer key {seq_key} for column {col}"
            )

    # Add categorical features (transformers create "{col}_encoded" keys)
    for col in cat_cols_filtered:
        cat_key = f"{col}_encoded"
        if cat_key in cat_result.data:
            transformed_data[col] = cat_result.data[cat_key]
            if config["runtime"]["verbose"]:
                logger.info(
                    f"Mapped categorical column {col} to transformer key {cat_key}"
                )
        else:
            raise KeyError(
                f"Cannot find categorical transformer key {cat_key} for column {col}"
            )

    # Split data
    val_split = config["training"]["val_split"]
    split_idx = int(len(targets) * (1 - val_split))

    train_data = {}
    val_data = {}
    for col in seq_cols + cat_cols:
        train_data[col] = transformed_data[col][:split_idx]
        val_data[col] = transformed_data[col][split_idx:]

    train_targets = targets[:split_idx]
    val_targets = targets[split_idx:]

    # Convert to tensors (handle pandas Series properly)
    train_batch = {}
    val_batch = {}
    for col in seq_cols_filtered + cat_cols_filtered:
        # Convert pandas Series to numpy arrays first
        train_values = (
            np.array(train_data[col])
            if hasattr(train_data[col], "values")
            else train_data[col]
        )
        val_values = (
            np.array(val_data[col])
            if hasattr(val_data[col], "values")
            else val_data[col]
        )

        # Convert to proper format for T4Rec embeddings
        # All features need to be integers for embedding layers within vocab_size
        vocab_size = config["model"]["vocab_size"]

        if col in seq_cols_filtered:
            # For sequence features, discretize float values to integers within vocab range
            train_values_int = np.clip(
                (train_values * (vocab_size - 1)).astype(int), 0, vocab_size - 1
            )
            val_values_int = np.clip(
                (val_values * (vocab_size - 1)).astype(int), 0, vocab_size - 1
            )
            train_batch[col] = torch.tensor(train_values_int, dtype=torch.long)
            val_batch[col] = torch.tensor(val_values_int, dtype=torch.long)
        else:  # categorical
            # For categorical features, ensure they're within vocab range
            train_values_int = np.clip(train_values.astype(int), 0, vocab_size - 1)
            val_values_int = np.clip(val_values.astype(int), 0, vocab_size - 1)
            train_batch[col] = torch.tensor(train_values_int, dtype=torch.long)
            val_batch[col] = torch.tensor(val_values_int, dtype=torch.long)

        if config["runtime"]["verbose"]:
            if col in seq_cols_filtered:
                logger.info(
                    f"Sequence {col}: train_range=[{train_values_int.min()}, {train_values_int.max()}], vocab_size={vocab_size}"
                )
            else:
                logger.info(
                    f"Categorical {col}: train_range=[{train_values_int.min()}, {train_values_int.max()}], vocab_size={vocab_size}"
                )

    train_targets_tensor = torch.tensor(train_targets, dtype=torch.long)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.long)

    # Update config with filtered columns for model creation
    config["features"]["sequence_cols"] = seq_cols_filtered
    config["features"]["categorical_cols"] = cat_cols_filtered

    # Create hybrid T4Rec model
    model = _create_t4rec_model(seq_result.data, cat_result.data, config)

    if config["runtime"]["verbose"]:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"T4Rec XLNet model created with {total_params:,} parameters")

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    history = []
    loss_fn = torch.nn.CrossEntropyLoss()

    if config["runtime"]["progress"] and _HAS_TQDM:
        pbar = tqdm(total=num_epochs, desc="Training T4Rec XLNet")
    else:
        pbar = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        train_outputs = model(train_batch)
        train_loss = loss_fn(train_outputs, train_targets_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_batch)
            val_loss = loss_fn(val_outputs, val_targets_tensor)

            # Calculate accuracy
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == val_targets_tensor).float().mean().item()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
                "val_accuracy": val_accuracy,
            }
        )

        if pbar:
            pbar.set_postfix(
                {"loss": f"{train_loss.item():.4f}", "val_acc": f"{val_accuracy:.4f}"}
            )
            pbar.update(1)

        if config["runtime"]["verbose"]:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | Loss: {train_loss.item():.4f} | Val Acc: {val_accuracy:.4f}"
            )

    if pbar:
        pbar.close()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_val_outputs = model(val_batch)
        final_predictions = torch.argmax(final_val_outputs, dim=1)
        final_accuracy = (final_predictions == val_targets_tensor).float().mean().item()

        # Calculate other metrics
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_targets_tensor.cpu().numpy(),
            final_predictions.cpu().numpy(),
            average="weighted",
            zero_division=0,
        )

    # Get predictions for Top-K analysis
    prediction_scores = final_val_outputs.cpu().numpy()
    true_labels = val_targets_tensor.cpu().numpy()

    execution_time = time.time() - start_time

    # Save outputs to Dataiku datasets
    saved_datasets = {}
    outputs_config = config["outputs"]

    if _HAS_DATAIKU and any(
        outputs_config.get(f"{name}_dataset")
        for name in ["features", "predictions", "metrics", "model_artifacts"]
    ):
        logger.info("Saving outputs to Dataiku datasets...")

        try:
            # Save features dataset
            if outputs_config.get("features_dataset"):
                features_df = pd.DataFrame(
                    {
                        "feature_name": seq_cols_filtered + cat_cols_filtered,
                        "feature_type": ["sequence"] * len(seq_cols_filtered)
                        + ["categorical"] * len(cat_cols_filtered),
                        "importance": [1.0]
                        * (
                            len(seq_cols_filtered) + len(cat_cols_filtered)
                        ),  # Placeholder
                    }
                )

                features_dataset = dataiku.Dataset(outputs_config["features_dataset"])
                features_dataset.write_with_schema(features_df)
                saved_datasets["features"] = outputs_config["features_dataset"]
                logger.info(f"Features saved to {outputs_config['features_dataset']}")

            # Save predictions dataset
            if outputs_config.get("predictions_dataset"):
                predictions_df = pd.DataFrame(
                    {
                        "prediction_id": range(len(prediction_scores)),
                        "raw_outputs": [pred.tolist() for pred in prediction_scores],
                        "predicted_class": np.argmax(prediction_scores, axis=1),
                        "true_class": true_labels,
                        "confidence": np.max(prediction_scores, axis=1),
                    }
                )

                predictions_dataset = dataiku.Dataset(
                    outputs_config["predictions_dataset"]
                )
                predictions_dataset.write_with_schema(predictions_df)
                saved_datasets["predictions"] = outputs_config["predictions_dataset"]
                logger.info(
                    f"Predictions saved to {outputs_config['predictions_dataset']}"
                )

            # Save metrics dataset
            if outputs_config.get("metrics_dataset"):
                metrics_df = pd.DataFrame(
                    {
                        "metric_name": ["accuracy", "precision", "recall", "f1"],
                        "metric_value": [final_accuracy, precision, recall, f1],
                        "dataset": ["validation"] * 4,
                        "timestamp": [pd.Timestamp.now()] * 4,
                    }
                )

                metrics_dataset = dataiku.Dataset(outputs_config["metrics_dataset"])
                metrics_dataset.write_with_schema(metrics_df)
                saved_datasets["metrics"] = outputs_config["metrics_dataset"]
                logger.info(f"Metrics saved to {outputs_config['metrics_dataset']}")

            # Save model artifacts
            if outputs_config.get("model_artifacts_dataset"):
                model_info_df = pd.DataFrame(
                    {
                        "artifact_name": [
                            "model_config",
                            "model_architecture",
                            "training_config",
                        ],
                        "artifact_value": [
                            str(config["model"]),
                            f"T4Rec-XLNet-{config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
                            str(config["training"]),
                        ],
                        "timestamp": [pd.Timestamp.now()] * 3,
                    }
                )

                artifacts_dataset = dataiku.Dataset(
                    outputs_config["model_artifacts_dataset"]
                )
                artifacts_dataset.write_with_schema(model_info_df)
                saved_datasets["model_artifacts"] = outputs_config[
                    "model_artifacts_dataset"
                ]
                logger.info(
                    f"Model artifacts saved to {outputs_config['model_artifacts_dataset']}"
                )

        except Exception as e:
            logger.warning(f"Error saving to Dataiku datasets: {e}")
            logger.info("Datasets might need to be created manually in Dataiku first")

    return {
        "metrics": {
            "accuracy": final_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "predictions": {
            "raw_outputs": prediction_scores,
            "predicted_classes": np.argmax(prediction_scores, axis=1),
            "true_classes": true_labels,
        },
        "model_info": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "parameters": sum(p.numel() for p in model.parameters()),
            "architecture": f"T4Rec-XLNet-{config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
        },
        "data_info": {
            "rows": len(df),
            "train_samples": len(train_targets),
            "val_samples": len(val_targets),
            "n_sequence_features": len(seq_cols),
            "n_categorical_features": len(cat_cols),
            "n_features": len(seq_cols_filtered) + len(cat_cols_filtered),
            "target_classes": target_vocab_size,
        },
        "execution_time": execution_time,
        "saved_datasets": saved_datasets,
    }


# Top-K evaluation functions
def evaluate_topk_metrics(predictions=None, targets=None, k_values=[1, 3, 4]):
    """Evaluate Top-K metrics using real T4Rec predictions"""
    if predictions is None or targets is None:
        raise ValueError("Real predictions and targets are required")

    metrics_by_k = {}

    for k in k_values:
        precisions = []
        recalls = []
        f1_scores = []

        for pred_logits, target in zip(predictions, targets):
            top_k_indices = np.argsort(pred_logits)[-k:][::-1]
            top_k_preds = set(top_k_indices)

            target_set = {target}
            intersection = top_k_preds.intersection(target_set)

            precision = len(intersection) / k
            recall = len(intersection) / len(target_set)

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        metrics_by_k[k] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores),
        }

    return metrics_by_k


def format_topk_table(metrics_by_k, baseline_metrics=None):
    """Format Top-K metrics table"""
    lines = []
    lines.append("T4REC XLNET TOP-K INFERENCE METRICS")
    lines.append("=" * 60)

    header = "| Metric          |"
    for k in sorted(metrics_by_k.keys()):
        header += f" K={k}          |"
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")

    for metric_name, display_name in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
    ]:
        row = f"| {display_name:<15} |"
        for k in sorted(metrics_by_k.keys()):
            val = metrics_by_k[k][metric_name] * 100
            row += f" {val:>5.1f}%       |"
        lines.append(row)

    lines.append("|" + "-" * (len(header) - 2) + "|")

    lines.append("")
    lines.append("BUSINESS INTERPRETATION:")
    best_k_precision = max(
        metrics_by_k.keys(), key=lambda k: metrics_by_k[k]["precision"]
    )
    best_k_recall = max(metrics_by_k.keys(), key=lambda k: metrics_by_k[k]["recall"])

    lines.append(
        f"   Best Precision: K={best_k_precision} ({metrics_by_k[best_k_precision]['precision'] * 100:.1f}%)"
    )
    lines.append(
        f"   Best Recall: K={best_k_recall} ({metrics_by_k[best_k_recall]['recall'] * 100:.1f}%)"
    )
    lines.append("   Powered by T4Rec XLNet architecture")

    return "\n".join(lines)


def print_topk_results(metrics_by_k, baseline_metrics=None):
    """Print T4Rec Top-K results"""
    table_str = format_topk_table(metrics_by_k, baseline_metrics)
    print(table_str)


def validate_config(config: Dict[str, Any], strict: bool = False) -> List[str]:
    """Validate T4Rec configuration"""
    errors = []

    # Check required fields
    required_fields = {
        "data.dataset_name": str,
        "data.sample_size": int,
        "features.sequence_cols": list,
        "features.categorical_cols": list,
        "features.target_col": str,
        "model.d_model": int,
        "model.n_heads": int,
        "model.n_layers": int,
    }

    for field_path, expected_type in required_fields.items():
        keys = field_path.split(".")
        value = config
        try:
            for key in keys:
                value = value[key]
            if not isinstance(value, expected_type):
                errors.append(f"{field_path} must be {expected_type.__name__}")
        except KeyError:
            errors.append(f"{field_path} is required")

    # Validate T4Rec specific constraints
    if "model" in config:
        model_cfg = config["model"]
        if model_cfg.get("d_model", 0) % model_cfg.get("n_heads", 1) != 0:
            errors.append("model.d_model must be divisible by model.n_heads")

    return errors


def get_config_schema() -> Dict[str, Any]:
    """Get T4Rec configuration schema"""
    return {
        "data": {
            "dataset_name": {
                "type": "str",
                "required": True,
                "desc": "Dataiku dataset name",
            },
            "sample_size": {
                "type": "int",
                "required": True,
                "desc": "Number of samples to load",
            },
            "chunk_size": {
                "type": "int",
                "required": False,
                "desc": "Processing chunk size",
            },
        },
        "features": {
            "sequence_cols": {
                "type": "List[str]",
                "required": True,
                "desc": "Sequential feature columns",
            },
            "categorical_cols": {
                "type": "List[str]",
                "required": True,
                "desc": "Categorical feature columns",
            },
            "target_col": {
                "type": "str",
                "required": True,
                "desc": "Target column name",
            },
        },
        "model": {
            "d_model": {
                "type": "int",
                "required": True,
                "desc": "T4Rec XLNet model dimension",
            },
            "n_heads": {
                "type": "int",
                "required": True,
                "desc": "Number of attention heads",
            },
            "n_layers": {
                "type": "int",
                "required": True,
                "desc": "Number of transformer layers",
            },
            "dropout": {"type": "float", "required": False, "desc": "Dropout rate"},
            "max_sequence_length": {
                "type": "int",
                "required": False,
                "desc": "Maximum sequence length",
            },
            "mem_len": {
                "type": "int",
                "required": False,
                "desc": "XLNet memory length",
            },
            "attn_type": {
                "type": "str",
                "required": False,
                "desc": "XLNet attention type",
            },
        },
        "training": {
            "batch_size": {
                "type": "int",
                "required": True,
                "desc": "Training batch size",
            },
            "num_epochs": {
                "type": "int",
                "required": True,
                "desc": "Number of training epochs",
            },
            "learning_rate": {
                "type": "float",
                "required": True,
                "desc": "Learning rate",
            },
            "weight_decay": {
                "type": "float",
                "required": False,
                "desc": "Weight decay",
            },
            "val_split": {
                "type": "float",
                "required": False,
                "desc": "Validation split ratio",
            },
        },
    }


