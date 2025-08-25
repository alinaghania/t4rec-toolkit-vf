from .registry import ModelRegistry, get_available_models, create_model
from .base_builder import BaseModelBuilder
from .xlnet_builder import XLNetModelBuilder
from .gpt2_builder import GPT2ModelBuilder
from .t4rec_advanced import T4RecAdvancedModel

# Registry global pré-configuré
registry = ModelRegistry()

__all__ = [
    "ModelRegistry",
    "BaseModelBuilder",
    "XLNetModelBuilder",
    "GPT2ModelBuilder",
    "T4RecAdvancedModel",
    "get_available_models",
    "create_model",
    "registry",
]
