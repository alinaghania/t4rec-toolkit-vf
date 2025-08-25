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
    from transformers4rec.torch.features.embedding import EmbeddingFeatures
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures
    from transformers4rec.torch.features.tabular import FeatureConfig, TableConfig
    from transformers4rec.torch.model.head import Head
    from transformers4rec.torch.model.prediction_task import (
        RegressionTask,
        BinaryClassificationTask,
        MultiClassClassificationTask,
    )

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


def _create_t4rec_schema(
    seq_cols: List[str], cat_cols: List[str], config: Dict[str, Any]
) -> tr.Schema:
    """Create T4Rec schema from features"""
    schema_config = []

    # Add sequence features
    for col in seq_cols:
        schema_config.append(
            FeatureConfig(
                name=col,
                dtype="float32",
                tags=["continuous", "sequence"],
                cardinality=None,
                min_val=None,
                max_val=None,
            )
        )

    # Add categorical features
    cat_config = config["transformers"]["categorical"]
    for col in cat_cols:
        schema_config.append(
            FeatureConfig(
                name=col,
                dtype="int64",
                tags=["categorical"],
                cardinality=cat_config["max_categories"],
                min_val=0,
                max_val=cat_config["max_categories"] - 1,
            )
        )

    # Add target feature
    target_col = config["features"]["target_col"]
    schema_config.append(
        FeatureConfig(
            name=target_col,
            dtype="int64",
            tags=["target"],
            cardinality=config["model"]["vocab_size"],
            min_val=0,
            max_val=config["model"]["vocab_size"] - 1,
        )
    )

    return tr.Schema(schema_config)


def _create_t4rec_model(schema: tr.Schema, config: Dict[str, Any]) -> tr.Model:
    """Create pure T4Rec XLNet model"""
    model_config = config["model"]

    # Create input features
    input_features = tr.TabularSequenceFeatures.from_schema(
        schema=schema,
        max_sequence_length=model_config["max_sequence_length"],
        continuous_projection=model_config["d_model"],
        categorical_cardinalities="auto",
    )

    # Create XLNet body
    xlnet_config = tr.XLNetConfig(
        d_model=model_config["d_model"],
        n_head=model_config["n_heads"],
        n_layer=model_config["n_layers"],
        dropout=model_config["dropout"],
        mem_len=model_config["mem_len"],
        attn_type=model_config["attn_type"],
    )

    body = tr.XLNetBlock(xlnet_config)

    # Create prediction head
    target_col = config["features"]["target_col"]
    prediction_task = MultiClassClassificationTask(
        target_name=target_col,
        num_classes=config["model"]["vocab_size"],
        summary_type="last",
    )

    head = tr.Head(
        body=body,
        prediction_tasks=prediction_task,
    )

    # Create complete model
    model = tr.Model(
        input_module=input_features,
        prediction_module=head,
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

    # Transform features
    seq_transformer = SequenceTransformer()
    cat_transformer = CategoricalTransformer()

    seq_result = seq_transformer.fit(df[seq_cols]).transform(df[seq_cols])
    cat_result = cat_transformer.fit(df[cat_cols]).transform(df[cat_cols])

    # Prepare target
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    targets = encoder.fit_transform(df[target_col])
    target_vocab_size = len(encoder.classes_)
    config["model"]["vocab_size"] = target_vocab_size

    # Create T4Rec schema
    schema = _create_t4rec_schema(seq_cols, cat_cols, config)

    # Prepare data for T4Rec
    transformed_data = {}

    # Add sequence features
    for i, col in enumerate(seq_cols):
        transformed_data[col] = seq_result.data[col]

    # Add categorical features
    for i, col in enumerate(cat_cols):
        transformed_data[col] = cat_result.data[col]

    # Add target
    transformed_data[target_col] = targets

    # Convert to T4Rec dataset format
    dataset_df = pd.DataFrame(transformed_data)

    # Split data
    val_split = config["training"]["val_split"]
    split_idx = int(len(dataset_df) * (1 - val_split))
    train_df = dataset_df[:split_idx]
    val_df = dataset_df[split_idx:]

    # Create T4Rec datasets
    train_dataset = tr.TabularDataset.from_df(
        train_df,
        schema=schema,
        targets=[target_col],
    )

    val_dataset = tr.TabularDataset.from_df(
        val_df,
        schema=schema,
        targets=[target_col],
    )

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create T4Rec model
    model = _create_t4rec_model(schema, config)

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

    if config["runtime"]["progress"] and _HAS_TQDM:
        pbar = tqdm(total=num_epochs, desc="Training T4Rec XLNet")
    else:
        pbar = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.losses
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_metrics = model.evaluate(val_loader)
        val_accuracy = val_metrics.get("accuracy", 0.0)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if pbar:
            pbar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "val_acc": f"{val_accuracy:.4f}"}
            )
            pbar.update(1)

        if config["runtime"]["verbose"]:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}"
            )

    if pbar:
        pbar.close()

    # Final evaluation
    model.eval()
    final_metrics = model.evaluate(val_loader)

    # Get predictions for Top-K analysis
    predictions = model.predict(val_loader)
    prediction_scores = predictions.prediction_response
    true_labels = val_df[target_col].values

    execution_time = time.time() - start_time

    return {
        "metrics": {
            "accuracy": final_metrics.get("accuracy", 0.0),
            "precision": final_metrics.get("precision", 0.0),
            "recall": final_metrics.get("recall", 0.0),
            "f1": final_metrics.get("f1", 0.0),
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


