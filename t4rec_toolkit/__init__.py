"""
T4Rec Toolkit - Bibliothèque d'outils pour les modèles de recommandation T4Rec
"""

# Core components
from .core import (
    BaseTransformer,
    TransformationResult,
    DataValidator,
    ValidationResult,
    T4RecToolkitError,
    DataValidationError,
    TransformationError,
    SchemaError,
    ConfigurationError,
)

# Transformers
from .transformers import (
    SequenceTransformer,
    CategoricalTransformer,
    NumericalTransformer,
)

# Model builders
from .models import (
    ModelRegistry,
    BaseModelBuilder,
    XLNetModelBuilder,
    GPT2ModelBuilder,
    get_available_models,
    create_model,
    registry,
)

# Adapters
from .adapters import T4RecAdapter, DataikuAdapter

# Utils
from .utils import (
    save_model,
    load_model,
    save_config,
    load_config,
    merge_configs,
    validate_config_schema,
    get_default_training_args,
    adapt_config_to_environment,
    select_features_for_t4rec,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseTransformer",
    "TransformationResult",
    "DataValidator",
    "ValidationResult",
    "T4RecToolkitError",
    "DataValidationError",
    "TransformationError",
    "SchemaError",
    "ConfigurationError",
    # Transformers
    "SequenceTransformer",
    "CategoricalTransformer",
    "NumericalTransformer",
    # Models
    "ModelRegistry",
    "BaseModelBuilder",
    "XLNetModelBuilder",
    "GPT2ModelBuilder",
    "get_available_models",
    "create_model",
    "registry",
    # Adapters
    "T4RecAdapter",
    "DataikuAdapter",
    # Utils
    "save_model",
    "load_model",
    "save_config",
    "load_config",
    "merge_configs",
    "validate_config_schema",
    "get_default_training_args",
    "adapt_config_to_environment",
    "select_features_for_t4rec",
]

