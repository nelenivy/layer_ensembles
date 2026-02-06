"""
Universal embedding extraction from transformer models.

Supports any HuggingFace model with hidden state output.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

from src.models.pooling import get_pooling_function


logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract layer-wise embeddings from transformer models.

    Args:
        model_name: HuggingFace model identifier
        pooling_strategy: Pooling method ('mean', 'cls', 'max', 'last_token', 'pooler')
        device: Device to run on ('cuda', 'cpu', or None for auto)
        max_length: Maximum sequence length
        use_fp16: Use mixed precision (faster, less memory)
    """

    def __init__(
        self,
        model_name: str,
        pooling_strategy: str = 'mean',
        device: Optional[str] = None,
        max_length: int = 256,
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Pooling: {pooling_strategy}")
        logger.info(f"FP16: {use_fp16}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)

        # Enable FP16 if requested
        if use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()

        self.model.eval()

        # Get pooling function
        self.pooling_fn = get_pooling_function(pooling_strategy)

        # Model properties
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Dictionary mapping layer_idx -> embeddings tensor [num_texts, hidden_dim]
        """
        all_embeddings = {layer: [] for layer in range(self.num_layers + 1)}

        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting embeddings")
        else:
            iterator = list(iterator)

        with torch.no_grad():
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs)

                # Extract hidden states (tuple of [batch, seq_len, hidden_dim])
                hidden_states = outputs.hidden_states  # (num_layers+1) tensors
                attention_mask = inputs['attention_mask']

                # Pool each layer
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    # Apply pooling
                    if self.pooling_strategy == 'pooler' and layer_idx == len(hidden_states) - 1:
                        # Use pooler output for last layer if available
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            pooled = outputs.pooler_output
                        else:
                            logger.warning("Pooler output not available, falling back to CLS")
                            pooled = layer_hidden[:, 0, :]
                    else:
                        pooled = self.pooling_fn(layer_hidden, attention_mask)

                    # Move to CPU and store
                    all_embeddings[layer_idx].append(pooled.cpu())

        # Concatenate all batches
        result = {}
        for layer_idx in range(self.num_layers + 1):
            result[layer_idx] = torch.cat(all_embeddings[layer_idx], dim=0)

        return result

    def get_layer_names(self) -> List[str]:
        """Get layer names for labeling."""
        return [f"Layer {i}" for i in range(self.num_layers + 1)]
