"""
Aggregated encoder that uses provided weights.
Inherits from MTEB's EncoderProtocol for full compatibility.
Includes pooler_output support for SimCSE models.
"""

import torch
import pickle
import numpy as np
from typing import Any, Optional, Dict, Union, List
from torch.utils.data import DataLoader

from mteb.models import EncoderProtocol  # ✅ CORRECT IMPORT FROM ORIGINAL


class SimpleWeightedAggregation:
    """Simple weighted aggregation that uses provided weights."""

    def __init__(self, weights: np.ndarray):
        """Initialize with weights."""
        self.weights = weights / weights.sum()  # Normalize
        self.weights_tensor = torch.from_numpy(self.weights).float()

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using weights.

        Args:
            layer_embeddings: (num_layers, batch_size, hidden_dim)

        Returns:
            aggregated: (batch_size, hidden_dim)
        """
        weights = self.weights_tensor.to(layer_embeddings.device).view(-1, 1, 1)
        return (layer_embeddings * weights).sum(dim=0)

    def get_weights(self) -> np.ndarray:
        """Get weights."""
        return self.weights.copy()

    def set_weights(self, weights: np.ndarray):
        """Set new weights."""
        self.weights = weights / weights.sum()
        self.weights_tensor = torch.from_numpy(self.weights).float()


class LayerEncoder:
    """
    Layer encoder that extracts all layers.
    Includes pooler_output support for SimCSE models.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        use_pooler_output: bool = False
    ):
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = device
        self.use_pooler_output = use_pooler_output

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()

        self.max_length = 512

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


class AggregatedEncoder(EncoderProtocol):  # ✅ CORRECT INHERITANCE
    """
    Aggregated encoder that USES the weights you set!
    Inherits from MTEB's EncoderProtocol for full compatibility.
    """

    def __init__(
        self,
        model_name: str,
        similarity_matrix_path: Optional[str] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        pooling: str = "mean",
        batch_size: int = 32,
        device: Optional[str] = None,
        aggregation_weights: Optional[np.ndarray] = None,
        normalize_weights: bool = False,  # Ignored, kept for compatibility
        use_pooler_output: bool = False
    ):
        """Initialize encoder."""
        if not model_name:
            raise ValueError("model_name is required!")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load similarity matrix (for reference, not used if weights provided)
        if similarity_matrix is not None:
            sim_matrix = similarity_matrix
        elif similarity_matrix_path is not None:
            with open(similarity_matrix_path, 'rb') as f:
                sim_matrix = pickle.load(f)
            sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
        else:
            raise ValueError("Must provide similarity_matrix or similarity_matrix_path")

        self.pooling = pooling
        self.batch_size = batch_size
        self.num_layers = sim_matrix.shape[0]
        self.use_pooler_output = use_pooler_output

        # Create layer encoder
        self.encoder = LayerEncoder(
            model_name=self.model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=self.device,
            use_pooler_output=use_pooler_output
        )
        self.hidden_size = self.encoder.model.config.hidden_size

        # CRITICAL: Use simple aggregation with weights
        if aggregation_weights is not None:
            initial_weights = aggregation_weights
        else:
            # Uniform weights by default
            initial_weights = np.ones(self.num_layers) / self.num_layers

        self.aggregator = SimpleWeightedAggregation(initial_weights)

    # ========== MTEB Interface Methods ==========

    def encode(
        self,
        sentences: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences (MTEB interface)."""
        return self._encode_impl(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(
        self,
        queries: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self._encode_impl(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]], DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        # Handle dict format (MTEB corpus)
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                # Try multiple keys
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self._encode_impl(sentences, batch_size=batch_size, **kwargs)

        return self._encode_impl(corpus, batch_size=batch_size, **kwargs)

    def similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between queries and corpus."""
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ corpus_norm.T

    def similarity_pairwise(self, sentences1: np.ndarray, sentences2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        s1_norm = sentences1 / (np.linalg.norm(sentences1, axis=1, keepdims=True) + 1e-8)
        s2_norm = sentences2 / (np.linalg.norm(sentences2, axis=1, keepdims=True) + 1e-8)
        return np.sum(s1_norm * s2_norm, axis=1)

    # ========== Internal Implementation ==========

    def _encode_impl(
        self,
        sentences: Union[List[str], str, DataLoader],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Internal encoding implementation."""
        # Handle DataLoader
        if isinstance(sentences, DataLoader):
            all_embeddings = []
            for batch in sentences:
                batch_sentences = self._extract_sentences_from_batch(batch)
                if not batch_sentences:
                    continue
                batch_embs = self._encode_batch(batch_sentences)
                if batch_embs is not None and len(batch_embs) > 0:
                    all_embeddings.append(batch_embs)
            if not all_embeddings:
                return np.zeros((0, self.hidden_size), dtype=np.float32)
            return np.vstack(all_embeddings)

        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences or len(sentences) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            if not batch:
                continue

            batch_embs = self._encode_batch(batch)
            if batch_embs is not None and len(batch_embs) > 0:
                all_embeddings.append(batch_embs)

        if not all_embeddings:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)

    def _extract_sentences_from_batch(self, batch):
        """Extract sentences from batch (for DataLoader compatibility)."""
        if isinstance(batch, dict):
            for key in ["text", "sentence", "sentences", "query", "passage", "title", "content"]:
                if key in batch and batch[key] is not None:
                    return batch[key]
            for v in batch.values():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                    return v
        elif isinstance(batch, (list, tuple)):
            return batch
        else:
            return [str(batch)]
        return []

    def _encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode a single batch."""
        if not sentences or len(sentences) == 0:
            return None

        # Filter empty sentences
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        if not sentences:
            return None

        try:
            # Tokenize
            encoded = self.encoder.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.encoder.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.encoder.model(**encoded)

                # Extract all layers
                all_layer_embs = []
                for layer_idx in range(self.num_layers):
                    hidden_states = outputs.hidden_states[layer_idx]
                    pooled = self.encoder.pool(hidden_states, encoded['attention_mask'])

                    if pooled is None:
                        pooled = torch.zeros(len(sentences), self.hidden_size, device=self.device)

                    all_layer_embs.append(pooled)

                # Add pooler_output if requested
                if self.use_pooler_output and outputs.pooler_output is not None:
                    all_layer_embs.append(outputs.pooler_output)

                # Stack: (num_layers, B, H)
                all_layer_embs = torch.stack(all_layer_embs, dim=0)

                # Aggregate using weights
                result = self.aggregator.aggregate(all_layer_embs)

                if result is None:
                    return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)

                if isinstance(result, torch.Tensor):
                    result = result.cpu().numpy()

                return result

        except Exception as e:
            print(f"ERROR in _encode_batch: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)

    # ========== Weight Management ==========

    def get_aggregation_weights(self) -> np.ndarray:
        """Get current aggregation weights."""
        return self.aggregator.get_weights()

    def set_aggregation_weights(self, weights: np.ndarray):
        """Set new aggregation weights."""
        self.aggregator.set_weights(weights)

    def __repr__(self):
        return f"AggregatedEncoder(model={self.model_name}, pooling={self.pooling}, layers={self.num_layers})"
