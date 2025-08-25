"""
Configuration Manager for T4Rec Pipeline

Handles all configuration logic, validation, and schema management
"""

from typing import Dict, Any, List, Optional
import warnings


class T4RecConfig:
    """T4Rec configuration defaults based on dataset size"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_sequence_length: int,
        vocab_size: int,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size


def _dynamic_t4rec_defaults(sample_size: int) -> T4RecConfig:
    """Generate T4Rec defaults based on sample size"""
    if sample_size <= 5000:
        return T4RecConfig(
            d_model=256, n_heads=8, n_layers=3, max_sequence_length=12, vocab_size=500
        )
    elif sample_size <= 50000:
        return T4RecConfig(
            d_model=512, n_heads=16, n_layers=6, max_sequence_length=20, vocab_size=1000
        )
    else:
        return T4RecConfig(
            d_model=768, n_heads=24, n_layers=8, max_sequence_length=30, vocab_size=2000
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


def validate_config(config: Dict[str, Any], strict: bool = False) -> List[str]:
    """Validate T4Rec configuration and return errors"""
    errors = []

    # Required fields
    required_fields = [
        ("data", "dataset_name"),
        ("features", "target_col"),
        ("features", "sequence_cols"),
        ("features", "categorical_cols"),
    ]

    for section, field in required_fields:
        if section not in config:
            errors.append(f"Missing section: {section}")
            continue
        if field not in config[section] or not config[section][field]:
            errors.append(f"Missing required field: {section}.{field}")

    # Model validation
    if "model" in config:
        model_cfg = config["model"]

        # d_model % n_heads must be 0
        if model_cfg.get("d_model", 0) % model_cfg.get("n_heads", 1) != 0:
            errors.append(
                f"d_model ({model_cfg.get('d_model')}) must be divisible by "
                f"n_heads ({model_cfg.get('n_heads')})"
            )

        # Reasonable ranges
        if model_cfg.get("d_model", 0) < 64:
            errors.append("d_model should be >= 64")
        if model_cfg.get("n_heads", 0) < 1:
            errors.append("n_heads should be >= 1")

    # Training validation
    if "training" in config:
        train_cfg = config["training"]
        if train_cfg.get("val_split", 0) <= 0 or train_cfg.get("val_split", 1) >= 1:
            errors.append("val_split should be between 0 and 1")

    if strict and not errors:
        # Additional strict checks
        if len(config.get("features", {}).get("sequence_cols", [])) == 0:
            warnings.warn("No sequence columns specified")
        if len(config.get("features", {}).get("categorical_cols", [])) == 0:
            warnings.warn("No categorical columns specified")

    return errors


def get_config_schema() -> Dict[str, Any]:
    """Get configuration schema for documentation"""
    return {
        "data": {
            "dataset_name": "str - Name of Dataiku dataset",
            "sample_size": "int - Number of samples to load",
            "chunk_size": "int - Processing chunk size",
        },
        "features": {
            "sequence_cols": "List[str] - Sequence feature columns",
            "categorical_cols": "List[str] - Categorical feature columns",
            "target_col": "str - Target column name",
            "max_seq_features": "int|None - Max sequence features",
            "max_cat_features": "int|None - Max categorical features",
        },
        "model": {
            "d_model": "int - Model dimension (must be divisible by n_heads)",
            "n_heads": "int - Number of attention heads",
            "n_layers": "int - Number of transformer layers",
            "dropout": "float - Dropout rate",
            "max_sequence_length": "int - Maximum sequence length",
            "vocab_size": "int - Vocabulary size for embeddings",
        },
        "training": {
            "batch_size": "int - Training batch size",
            "num_epochs": "int - Number of training epochs",
            "learning_rate": "float - Learning rate",
            "weight_decay": "float - Weight decay for regularization",
            "val_split": "float - Validation split ratio (0-1)",
        },
        "runtime": {
            "verbose": "bool - Enable verbose logging",
            "progress": "bool - Show progress bars",
            "seed": "int - Random seed for reproducibility",
        },
        "outputs": {
            "features_dataset": "str|None - Output dataset for features",
            "predictions_dataset": "str|None - Output dataset for predictions",
            "metrics_dataset": "str|None - Output dataset for metrics",
            "local_dir": "str - Local output directory",
        },
    }


def print_config_help(format_type: str = "text") -> str:
    """Print configuration help"""
    schema = get_config_schema()

    if format_type == "text":
        help_text = "T4REC PIPELINE CONFIGURATION SCHEMA\n" + "=" * 50 + "\n\n"

        for section, fields in schema.items():
            help_text += f"[{section.upper()}]\n"
            for field, description in fields.items():
                help_text += f"  {field}: {description}\n"
            help_text += "\n"

        return help_text

    elif format_type == "dict":
        return schema

    else:
        raise ValueError(f"Unsupported format: {format_type}")
