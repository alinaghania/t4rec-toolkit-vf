"""
T4Rec Advanced Pipeline - Main Orchestrator

Simplified pipeline orchestrator that uses modular components:
- core.config_manager: Configuration management
- core.training_engine: Training and inference logic
- models.t4rec_advanced: Advanced T4Rec model architecture
"""

from typing import Dict, Any
import warnings

# Core components
from .core.config_manager import (
    blank_config,
    default_config,
    validate_config,
    get_config_schema,
    print_config_help,
)
from .core.training_engine import TrainingEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Export main functions for easy import
__all__ = [
    "blank_config",
    "default_config",
    "validate_config",
    "get_config_schema",
    "print_config_help",
    "run_training",
]


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run T4Rec advanced training pipeline

    Simplified orchestrator that delegates to TrainingEngine for all heavy lifting.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing metrics, model info, and training results

    Example:
        >>> config = default_config(sample_size=30000)
        >>> config["data"]["dataset_name"] = "MY_DATASET"
        >>> config["features"]["sequence_cols"] = ["col1", "col2"]
        >>> config["features"]["categorical_cols"] = ["cat1", "cat2"]
        >>> config["features"]["target_col"] = "target"
        >>> results = run_training(config)
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    """
    # Validate configuration
    errors = validate_config(config, strict=False)
    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    # Create and run training engine
    engine = TrainingEngine(config)
    results = engine.run_training()

    return results


# Legacy compatibility - will be deprecated
def print_config_help_legacy(format_type: str = "text") -> str:
    """Legacy function - use print_config_help instead"""
    warnings.warn(
        "print_config_help_legacy is deprecated, use print_config_help",
        DeprecationWarning,
    )
    return print_config_help(format_type)
