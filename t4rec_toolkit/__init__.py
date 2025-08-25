"""
T4Rec Toolkit - Bibliothèque d'outils pour les modèles de recommandation T4Rec
"""

# Core components
from .core import (
    BaseTransformer,
    TransformationResult,
    DataValidator,
    ValidationResult,
    SequenceValidator,
    T4RecToolkitError,
    DataValidationError,
    TransformationError,
    SchemaError,
    ConfigurationError,
    blank_config,
    default_config,
    validate_config,
    get_config_schema,
    print_config_help,
    TrainingEngine,
)

# Pipeline orchestrator
from .pipeline_core import run_training

# Transformers
from .transformers import (
    SequenceTransformer,
    CategoricalTransformer,
    NumericalTransformer,
)

# Model builders and advanced models
from .models import (
    ModelRegistry,
    BaseModelBuilder,
    XLNetModelBuilder,
    GPT2ModelBuilder,
    T4RecAdvancedModel,
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
    "SequenceValidator",
    "T4RecToolkitError",
    "DataValidationError",
    "TransformationError",
    "SchemaError",
    "ConfigurationError",
    # Configuration & Training
    "blank_config",
    "default_config",
    "validate_config",
    "get_config_schema",
    "print_config_help",
    "TrainingEngine",
    "run_training",
    # Transformers
    "SequenceTransformer",
    "CategoricalTransformer",
    "NumericalTransformer",
    # Models
    "ModelRegistry",
    "BaseModelBuilder",
    "XLNetModelBuilder",
    "GPT2ModelBuilder",
    "T4RecAdvancedModel",
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

