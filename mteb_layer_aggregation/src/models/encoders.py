"""
Base encoder classes with pooler_output support for SimCSE models.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel


class LayerEncoder:
    """
    Encoder that extracts all layer representations from a transformer model.
    Supports both hidden_states and pooler_output (for SimCSE models).
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cuda",
        use_pooler_output: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name
            pooling: 'mean' or 'cls'
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            device: Device to use
            use_pooler_output: Whether to use pooler_output for SimCSE models
        """
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_pooler_output = use_pooler_output

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()

        # Determine if model has pooler_output
        self.has_pooler = hasattr(self.model.config, 'use_pooler') or 'simcse' in model_name.lower()

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode sentences and return layer representations.

        Args:
            sentences: List of sentences or single sentence
            batch_size: Override default batch size

        Returns:
            embeddings: (N, num_layers, hidden_dim) for mean/cls pooling
                       or (N, num_layers+1, hidden_dim) if using pooler_output
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embs = self._encode_batch(batch)
            all_embeddings.append(batch_embs)

        return np.vstack(all_embeddings)

    def _encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode a single batch."""
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of (B, T, H)

        # Pool each layer
        all_layer_embs = []
        for layer_idx, layer_hidden in enumerate(hidden_states):
            pooled = self.pool(layer_hidden, inputs['attention_mask'])
            all_layer_embs.append(pooled)

        # Add pooler_output if available and requested
        if self.use_pooler_output and outputs.pooler_output is not None:
            all_layer_embs.append(outputs.pooler_output)

        # Stack: (B, num_layers, H)
        stacked = torch.stack(all_layer_embs, dim=1)

        return stacked.cpu().numpy()

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states.

        Args:
            hidden_states: (B, T, H)
            attention_mask: (B, T)

        Returns:
            pooled: (B, H)
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            masked_hidden = hidden_states * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            return sum_hidden / sum_mask
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


class SelectedLayersEncoder:
    """
    Encoder that aggregates selected layers with weights.
    Supports pooler_output for SimCSE models.
    """

    def __init__(
        self,
        model_name: str,
        layer_indices: List[int],
        layer_weights: np.ndarray,
        pooling: str = "mean",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cuda",
        pooler_idx: Optional[int] = None
    ):
        """
        Args:
            model_name: HuggingFace model name
            layer_indices: Which layers to use
            layer_weights: Weight for each selected layer
            pooling: 'mean' or 'cls'
            batch_size: Batch size
            max_length: Max sequence length
            device: Device to use
            pooler_idx: Layer index for pooler_output (e.g., 13 for 12-layer model)
                       If None, will be set to num_hidden_layers + 1
        """
        assert len(layer_indices) == len(layer_weights), "Mismatch in layers/weights length"

        self.model_name = model_name
        self.layer_indices = layer_indices
        self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()

        # Set pooler_idx
        if pooler_idx is None:
            pooler_idx = self.model.config.num_hidden_layers + 1  # 13 for 12-layer BERT
        self.pooler_idx = pooler_idx

        # Validate layer indices
        max_valid_idx = self.model.config.num_hidden_layers  # Max hidden state index
        for idx in layer_indices:
            if idx != self.pooler_idx and not (0 <= idx <= max_valid_idx):
                raise ValueError(f"Layer index {idx} out of range [0, {max_valid_idx}] or not pooler_idx")

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode sentences using weighted selected layers."""
        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embs = self._encode_batch(batch)
            all_embeddings.append(batch_embs)

        return np.vstack(all_embeddings)

    def _encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode a single batch."""
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        mask = inputs['attention_mask'].unsqueeze(-1).float()

        # Extract and weight selected layers
        layer_vecs = []
        for i, layer_idx in enumerate(self.layer_indices):
            if layer_idx == self.pooler_idx:
                # Use pooler_output
                if outputs.pooler_output is None:
                    raise RuntimeError(f"pooler_output not available, but pooler_idx={self.pooler_idx} requested")
                vec = outputs.pooler_output  # (B, H)
            else:
                # Use hidden_states[layer_idx]
                h = hidden_states[layer_idx]  # (B, T, H)

                if self.pooling == "cls":
                    vec = h[:, 0, :]  # CLS token
                elif self.pooling == "mean":
                    vec = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

            # Apply weight
            vec = vec * self.layer_weights[i].to(vec.device)
            layer_vecs.append(vec)

        # Sum weighted layers
        final_vec = torch.stack(layer_vecs, dim=0).sum(dim=0)  # (B, H)

        return final_vec.cpu().numpy()
