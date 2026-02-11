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
from mteb.models import EncoderProtocol  
from transformers import AutoTokenizer, AutoModel
from src.cache_manager import EmbeddingCache
from src.strategies import normalize_weights


class SimpleWeightedAggregation:
    """Simple weighted aggregation that uses provided weights."""

    def __init__(self, weights: np.ndarray):
        """Initialize with weights."""
        self.weights = normalize_weights(weights, threshold=0.001)
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
        self.weights = normalize_weights(weights, threshold=0.001)
        self.weights_tensor = torch.from_numpy(self.weights).float()

class LayerEncoder:
    """
    Layer encoder that extracts all layers with LMDB caching support.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        use_pooler_output: bool = False,
        use_cache: bool = False,  # NEW
        cache_dir: str = "./embedding_cache"  # NEW
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = device
        self.use_pooler_output = use_pooler_output
        self.use_cache = use_cache
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()
        
        self.max_length = 512
        self.num_layers = self.model.config.num_hidden_layers + 1
        
        # Initialize cache
        if use_cache:
            
            self.cache = EmbeddingCache(cache_dir)
            print(f"✓ LMDB cache enabled for LayerEncoder")
        else:
            self.cache = None
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def encode_batch(
        self,
        sentences: List[str],
        return_all_layers: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode a batch of sentences.
        
        Args:
            sentences: List of sentences to encode
            return_all_layers: If True, return list of arrays (one per layer)
                             If False, return single array (last layer only)
        
        Returns:
            If return_all_layers: List[np.ndarray] of shape [(B, H)] * num_layers
            Otherwise: np.ndarray of shape (B, H)
        """
        if not sentences:
            return [] if return_all_layers else np.zeros((0, self.model.config.hidden_size))
        
        # Normalize sentences
        sentences = [str(s).strip() for s in sentences if s]
        if not sentences:
            return [] if return_all_layers else np.zeros((0, self.model.config.hidden_size))
        
        # Check cache for each layer
        if self.use_cache and self.cache:
            cached_layers = self._get_cached_batch(sentences)
            if cached_layers is not None:
                # All layers cached
                if return_all_layers:
                    return cached_layers
                else:
                    return cached_layers[-1]  # Return last layer
            
            # Partial cache hit - for now, recompute all
            # (You could optimize this to only compute missing layers)
        
        # Compute embeddings
        try:
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
                
                # Extract all layers
                all_layer_embeddings = []
                for layer_idx in range(self.num_layers):
                    layer_hidden = hidden_states[layer_idx]
                    pooled = self.pool(layer_hidden, inputs['attention_mask'])
                    all_layer_embeddings.append(pooled.cpu().numpy())
                
                # Add pooler_output if requested
                if self.use_pooler_output and outputs.pooler_output is not None:
                    all_layer_embeddings.append(outputs.pooler_output.cpu().numpy())
            
            # Cache all layers
            if self.use_cache and self.cache:
                self._cache_batch(sentences, all_layer_embeddings)
            
            if return_all_layers:
                return all_layer_embeddings
            else:
                return all_layer_embeddings[-1]
        
        except Exception as e:
            print(f"Error in encode_batch: {e}")
            if return_all_layers:
                return [np.zeros((len(sentences), self.model.config.hidden_size)) 
                       for _ in range(self.num_layers)]
            else:
                return np.zeros((len(sentences), self.model.config.hidden_size))
    
    def _get_cached_batch(self, sentences: List[str]) -> Optional[List[np.ndarray]]:
        """
        Try to get all layers for all sentences from cache.
        Returns None if any sentence/layer is missing.
        """
        # Check if all sentences have all layers cached
        all_cached = []
        
        for layer_idx in range(self.num_layers):
            layer_embeddings = []
            for sentence in sentences:
                emb = self.cache.get_sentence(
                    self.model_name, layer_idx, self.pooling, sentence
                )
                if emb is None:
                    return None  # Cache miss
                layer_embeddings.append(emb)
            
            all_cached.append(np.vstack(layer_embeddings))
        
        return all_cached
    
    def _cache_batch(self, sentences: List[str], all_layer_embeddings: List[np.ndarray]):
        """Cache all layers for all sentences"""
        for layer_idx, layer_embs in enumerate(all_layer_embeddings):
            for sentence, emb in zip(sentences, layer_embs):
                self.cache.set_sentence(
                    self.model_name, layer_idx, self.pooling, sentence, emb
                )
    
    def __del__(self):
        """Close cache on cleanup"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.close()


class AggregatedEncoder(EncoderProtocol):
    """
    Aggregated encoder with caching support.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: Optional[str] = None,
        aggregation_weights: Optional[np.ndarray] = None,
        normalize_weights: bool = False,
        use_pooler_output: bool = False,
        use_cache: bool = False,
        cache_dir: str = "./embedding_cache"
    ):
        """Initialize encoder with caching support."""
        if not model_name:
            raise ValueError("model_name is required!")

        if aggregation_weights is not None:
            initial_weights = aggregation_weights
        else:
            initial_weights = np.ones(self.num_layers)
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pooling = pooling
        self.batch_size = batch_size
        self.num_layers = initial_weights.shape[0]
        self.use_pooler_output = use_pooler_output
        
        # Create layer encoder WITH CACHING
        self.encoder = LayerEncoder(
            model_name=self.model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=self.device,
            use_pooler_output=use_pooler_output,
            use_cache=use_cache,  # Pass through
            cache_dir=cache_dir   # Pass through
        )
        
        self.hidden_size = self.encoder.model.config.hidden_size
        
        # Setup aggregation weights

        self.aggregator = SimpleWeightedAggregation(initial_weights)
    
    def _encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode a single batch using cached layer encoder."""
        if not sentences or len(sentences) == 0:
            return None
        
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        
        if not sentences:
            return None
        
        try:
            # Get all layer embeddings (uses cache internally)
            all_layer_embs = self.encoder.encode_batch(
                sentences, return_all_layers=True
            )
            
            if not all_layer_embs:
                return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)
            
            # Stack: (num_layers, B, H)
            all_layer_embs = [
                torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
                for emb in all_layer_embs
            ]
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

    # ========== Weight Management ==========

    def get_aggregation_weights(self) -> np.ndarray:
        """Get current aggregation weights."""
        return self.aggregator.get_weights()

    def set_aggregation_weights(self, weights: np.ndarray):
        """Set new aggregation weights."""
        self.aggregator.set_weights(weights)

    def __repr__(self):
        return f"AggregatedEncoder(model={self.model_name}, pooling={self.pooling}, layers={self.num_layers})"
        
