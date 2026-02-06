"""
Model configuration and registry.
"""

from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for a transformer model."""

    name: str
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    max_length: int = 512
    has_pooler: bool = False

    def __post_init__(self):
        """Infer properties if not provided."""
        if self.hidden_size is None or self.num_layers is None:
            self._infer_from_model()

    def _infer_from_model(self):
        """Load model config to infer properties."""
        try:
            model = AutoModel.from_pretrained(self.name, output_hidden_states=True)
            config = model.config

            self.hidden_size = config.hidden_size

            # Try different attribute names for number of layers
            if hasattr(config, 'num_hidden_layers'):
                self.num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layers'):
                self.num_layers = config.n_layers
            elif hasattr(config, 'num_layers'):
                self.num_layers = config.num_layers
            else:
                raise AttributeError(f"Cannot determine number of layers for {self.name}")

            # Check if model has pooler
            self.has_pooler = hasattr(model, 'pooler') and model.pooler is not None

        except Exception as e:
            raise ValueError(f"Failed to load model config for {self.name}: {e}")


def get_model_config(model_name: str, **kwargs) -> ModelConfig:
    """
    Get model configuration.

    Args:
        model_name: HuggingFace model identifier
        **kwargs: Additional config parameters

    Returns:
        ModelConfig instance
    """
    return ModelConfig(name=model_name, **kwargs)
