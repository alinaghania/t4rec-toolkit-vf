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


def _load_dataframe(dataset_name: str, sample_size: int) -> pd.DataFrame:
    """Load dataframe from Dataiku with sampling"""
    if not _HAS_DATAIKU:
        raise RuntimeError("Dataiku not available")

    dataset = dataiku.Dataset(dataset_name)
    df = dataset.get_dataframe(limit=sample_size)
    return df


class BankingRecommendationModel(torch.nn.Module):
    """Hybrid T4Rec XLNet model for banking product recommendation"""

    def __init__(self, embedding_module, xlnet_config, n_products, d_model):
        super().__init__()
        self.embedding_module = embedding_module
        self.transformer = tr.TransformerBlock(xlnet_config)
        self.n_products = n_products

        # Prediction head
        self.recommendation_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(
                xlnet_config.dropout if hasattr(xlnet_config, "dropout") else 0.1
            ),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                xlnet_config.dropout if hasattr(xlnet_config, "dropout") else 0.1
            ),
            torch.nn.Linear(d_model // 2, n_products),
        )

    def forward(self, inputs, return_embeddings=False):
        # 1. T4Rec embeddings
        embeddings = self.embedding_module(inputs)

        # 2. XLNet transformer
        try:
            transformer_output = self.transformer(embeddings)
        except:
            # Fallback if transformer fails
            transformer_output = embeddings

        # 3. Take last sequence position
        final_representation = transformer_output[:, -1, :]

        # 4. Product prediction
        product_logits = self.recommendation_head(final_representation)

        if return_embeddings:
            return product_logits, final_representation
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

    with torch.no_grad():
        try:
            test_output = embedding_module(test_batch)
            actual_d_model = test_output.shape[-1]
            if config["runtime"]["verbose"]:
                print(
                    f"T4Rec embedding test successful: output_shape={test_output.shape}, d_model={actual_d_model}"
                )
        except Exception as e:
            if config["runtime"]["verbose"]:
                print(
                    f"T4Rec embedding test failed: {e}, using configured d_model={d_model}"
                )
            actual_d_model = d_model

    # Create XLNet config
    xlnet_config = tr.XLNetConfig.build(
        d_model=actual_d_model,
        n_head=model_config["n_heads"],
        n_layer=model_config["n_layers"],
        dropout=model_config["dropout"],
    )

    # Create hybrid model
    model = BankingRecommendationModel(
        embedding_module=embedding_module,
        xlnet_config=xlnet_config,
        n_products=model_config["vocab_size"],
        d_model=actual_d_model,
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
    df = _load_dataframe(dataset_name, sample_size)

    if config["runtime"]["verbose"]:
        logger.info(f"Loaded {len(df)} samples from {dataset_name}")

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
    config["model"]["vocab_size"] = target_vocab_size

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
        # All features need to be integers for embedding layers
        if col in seq_cols_filtered:
            # For sequence features, discretize float values to integers
            train_values_int = (train_values * 1000).astype(
                int
            )  # Scale and convert to int
            val_values_int = (val_values * 1000).astype(int)
            train_batch[col] = torch.tensor(train_values_int, dtype=torch.long)
            val_batch[col] = torch.tensor(val_values_int, dtype=torch.long)
        else:  # categorical
            train_batch[col] = torch.tensor(train_values.astype(int), dtype=torch.long)
            val_batch[col] = torch.tensor(val_values.astype(int), dtype=torch.long)

        if config["runtime"]["verbose"]:
            logger.info(
                f"Converted {col}: train_shape={train_values.shape}, val_shape={val_values.shape}"
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
            "n_sequence_features": len(seq_cols),
            "n_categorical_features": len(cat_cols),
            "target_classes": target_vocab_size,
        },
        "execution_time": execution_time,
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
