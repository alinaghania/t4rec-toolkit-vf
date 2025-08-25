from .base_transformer import BaseTransformer, TransformationResult
from .validator import DataValidator, ValidationResult, SequenceValidator
from .exceptions import (
    T4RecToolkitError,
    DataValidationError,
    TransformationError,
    SchemaError,
    ConfigurationError,
)
from .config_manager import (
    blank_config,
    default_config,
    validate_config,
    get_config_schema,
    print_config_help,
)

# Conditional import for TrainingEngine (requires torch)
try:
    from .training_engine import TrainingEngine

    _HAS_TRAINING_ENGINE = True
except ImportError as e:
    TrainingEngine = None
    _HAS_TRAINING_ENGINE = False
    import warnings

    warnings.warn(f"TrainingEngine not available: {e}", ImportWarning)

# Base exports (always available)
_base_exports = [
    "BaseTransformer",
    "TransformationResult",
    "DataValidator",
    "ValidationResult",
    "SequenceValidator",
    "T4RecToolkitError",
    "DataValidationError",
    "TransformationError",
    "SchemaError",
    "ConfigurationError",
    "blank_config",
    "default_config",
    "validate_config",
    "get_config_schema",
    "print_config_help",
]

# Add TrainingEngine if available
__all__ = _base_exports + (["TrainingEngine"] if _HAS_TRAINING_ENGINE else [])
