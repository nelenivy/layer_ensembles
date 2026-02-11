# pca_encoders-2.py - Refactored with shared caching base class

import torch
import numpy as np
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, AutoModel
from mteb.models import EncoderProtocol
from abc import ABC, abstractmethod
from src.cache_manager import EmbeddingCache

class CachedLayerEncoderBase(ABC):
    """
    Base class for encoders that need cached layer embeddings.
    Handles all caching logic to avoid duplication.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling: str,
        batch_size: int,
        device: str,
        use_cache: bool,
        cache_dir: str,
        pooler_idx: Optional[int] = None
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()
        
        # Set pooler_idx
        if pooler_idx is None:
            pooler_idx = self.model.config.num_hidden_layers + 1
        self.pooler_idx = pooler_idx
        
        # Initialize cache
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)
        else:
            self.cache = None
    
    @abstractmethod
    def get_required_layers(self) -> List[int]:
        """Return list of layer indices this encoder needs"""
        pass
    
    def _get_layer_embeddings_cached(self, sentences: List[str]) -> Dict[int, np.ndarray]:
        """
        Get layer embeddings with caching.
        Returns dict: {layer_idx: embeddings_array of shape (batch_size, hidden_dim)}
        """
        required_layers = self.get_required_layers()
        cached_layers = {layer_idx: [] for layer_idx in required_layers}
        uncached_indices = []
        uncached_sentences = []
        
        # Try to get from cache
        if self.cache:
            for i, sentence in enumerate(sentences):
                all_cached = True
                for layer_idx in required_layers:
                    emb = self.cache.get_sentence(
                        self.model_name, layer_idx, self.pooling, sentence
                    )
                    if emb is None:
                        all_cached = False
                        break
                    cached_layers[layer_idx].append(emb)
                
                if not all_cached:
                    # Need to compute this sentence
                    uncached_indices.append(i)
                    uncached_sentences.append(sentence)
                    # Clear partial cache for this sentence
                    for layer_idx in required_layers:
                        if len(cached_layers[layer_idx]) > len(uncached_indices) - 1:
                            cached_layers[layer_idx].pop()
        else:
            uncached_indices = list(range(len(sentences)))
            uncached_sentences = sentences
        
        # Compute uncached embeddings
        if uncached_sentences:
            computed_layers = self._compute_layer_embeddings(uncached_sentences)
            
            # Merge computed with cached
            for layer_idx in required_layers:
                if not cached_layers[layer_idx]:
                    # All uncached
                    cached_layers[layer_idx] = computed_layers[layer_idx]
                else:
                    # Interleave cached and computed
                    result = []
                    cached_iter = iter(cached_layers[layer_idx])
                    computed_iter = iter(computed_layers[layer_idx])
                    
                    for i in range(len(sentences)):
                        if i in uncached_indices:
                            result.append(next(computed_iter))
                        else:
                            result.append(next(cached_iter))
                    
                    cached_layers[layer_idx] = result
        
        # Convert lists to arrays
        return {
            layer_idx: np.vstack(embs) if embs else np.zeros((0, self.model.config.hidden_size))
            for layer_idx, embs in cached_layers.items()
        }
    
    def _compute_layer_embeddings(self, sentences: List[str]) -> Dict[int, List[np.ndarray]]:
        """Compute layer embeddings without cache and save to cache"""
        required_layers = self.get_required_layers()
        
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            
            layer_embeddings = {}
            
            for layer_idx in required_layers:
                if layer_idx == self.pooler_idx:
                    # Use pooler_output
                    if outputs.pooler_output is None:
                        raise RuntimeError(f"pooler_output not available")
                    pooled = outputs.pooler_output
                else:
                    # Use hidden_states[layer_idx]
                    layer_hidden = hidden_states[layer_idx]
                    if self.pooling == "cls":
                        pooled = layer_hidden[:, 0, :]
                    else:  # mean
                        mask = inputs['attention_mask'].unsqueeze(-1).float()
                        pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                
                embeddings_np = pooled.cpu().numpy()
                layer_embeddings[layer_idx] = [embeddings_np[i] for i in range(len(sentences))]
                
                # Cache individual embeddings
                if self.cache:
                    for sentence, emb in zip(sentences, embeddings_np):
                        self.cache.set_sentence(
                            self.model_name, layer_idx, self.pooling, sentence, emb
                        )
        
        return layer_embeddings
    
    def close_cache(self):
        """Close cache connection"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.close()


class SelectedLayersPCAEncoder(EncoderProtocol, CachedLayerEncoderBase):
    """
    QP-weighted layer selection → Concat with sqrt(weights) → PCA.
    """
    
    def __init__(
        self,
        model_name: str,
        layer_indices: List[int],
        layer_weights: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: Optional[np.ndarray] = None,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        pooler_idx: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = "./embedding_cache"
    ):
        # Initialize base class with cache support
        CachedLayerEncoderBase.__init__(
            self,
            model_name=model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
            pooler_idx=pooler_idx
        )
        
        self.layer_indices = layer_indices
        self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)
        self.weights_sqrt = torch.sqrt(self.layer_weights)
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        
        if use_cache:
            print(f"✓ Cache enabled for SelectedLayersPCAEncoder (layers: {layer_indices})")
        
        # Validate layer indices
        max_idx = self.model.config.num_hidden_layers
        for idx in layer_indices:
            if idx != self.pooler_idx and not (0 <= idx <= max_idx):
                raise ValueError(f"Layer index {idx} out of range")
    
    def get_required_layers(self) -> List[int]:
        """Return layers needed by this encoder"""
        return self.layer_indices
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using weighted concat → PCA with caching"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Get layer embeddings (uses cache from base class)
            layer_embeddings = self._get_layer_embeddings_cached(batch)
            
            # Apply sqrt(weight) and concatenate
            weighted_layers = []
            for idx, layer_idx in enumerate(self.layer_indices):
                layer_emb = torch.from_numpy(layer_embeddings[layer_idx]).to(self.device)
                weighted = layer_emb * self.weights_sqrt[idx].to(layer_emb.device)
                weighted_layers.append(weighted)
            
            # Concatenate weighted layers
            concat = torch.cat(weighted_layers, dim=-1)
            
            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)
            pca_emb = concat @ self.pca_components.T.to(concat.device)
            
            all_embeddings.append(pca_emb.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        return self.encode(queries, **kwargs)
    
    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)
    
    def __del__(self):
        self.close_cache()


class ClusterPCAEncoder(EncoderProtocol, CachedLayerEncoderBase):
    """
    Hierarchical clustering → Weighted cluster concat → PCA.
    """
    
    def __init__(
        self,
        model_name: str,
        clusters: List[List[int]],
        cluster_weights: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: Optional[np.ndarray] = None,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        pooler_idx: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = "./embedding_cache"
    ):
        # Initialize base class with cache support
        CachedLayerEncoderBase.__init__(
            self,
            model_name=model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
            pooler_idx=pooler_idx
        )
        
        self.clusters = clusters
        self.cluster_weights = torch.tensor(cluster_weights, dtype=torch.float32)
        self.weights_sqrt = torch.sqrt(self.cluster_weights)
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        
        # Get all unique layer indices from clusters
        self.all_layer_indices = sorted(set(
            layer_idx for cluster in clusters for layer_idx in cluster
        ))
        
        if use_cache:
            print(f"✓ Cache enabled for ClusterPCAEncoder ({len(clusters)} clusters)")
    
    def get_required_layers(self) -> List[int]:
        """Return all layers needed by clusters"""
        return self.all_layer_indices
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using cluster concat → PCA with caching"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Get layer embeddings (uses cache from base class)
            layer_embeddings = self._get_layer_embeddings_cached(batch)
            
            # Process each cluster
            cluster_vecs = []
            for cluster_idx, cluster_layers in enumerate(self.clusters):
                # Average layers within cluster
                layer_embs = [
                    torch.from_numpy(layer_embeddings[layer_idx]).to(self.device)
                    for layer_idx in cluster_layers
                ]
                cluster_avg = torch.stack(layer_embs, dim=0).mean(dim=0)
                
                # Apply sqrt(weight)
                weighted = cluster_avg * self.weights_sqrt[cluster_idx].to(cluster_avg.device)
                cluster_vecs.append(weighted)
            
            # Concatenate clusters
            concat = torch.cat(cluster_vecs, dim=-1)
            
            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)
            pca_emb = concat @ self.pca_components.T.to(concat.device)
            
            all_embeddings.append(pca_emb.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        return self.encode(queries, **kwargs)
    
    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)
    
    def __del__(self):
        self.close_cache()


class AllLayersPCAEncoder(EncoderProtocol, CachedLayerEncoderBase):
    """
    Concat ALL layers (no weighting) → PCA.
    Baseline method from notebooks.
    """
    
    def __init__(
        self,
        model_name: str,
        pca_components: np.ndarray,
        pca_mean: Optional[np.ndarray] = None,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        use_cache: bool = False,
        cache_dir: str = "./embedding_cache"
    ):
        # Initialize base class with cache support
        CachedLayerEncoderBase.__init__(
            self,
            model_name=model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
            pooler_idx=None  # Don't use pooler_output for baseline
        )
        
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        
        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        self.all_layer_indices = list(range(self.num_layers))
        
        if use_cache:
            print(f"✓ Cache enabled for AllLayersPCAEncoder ({self.num_layers} layers)")
    
    def get_required_layers(self) -> List[int]:
        """Return all layers (0 to num_layers-1)"""
        return self.all_layer_indices
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences by concatenating all layers → PCA with caching"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Get all layer embeddings (uses cache from base class)
            layer_embeddings = self._get_layer_embeddings_cached(batch)
            
            # Concatenate all layers (in order)
            all_layers = [
                torch.from_numpy(layer_embeddings[layer_idx]).to(self.device)
                for layer_idx in self.all_layer_indices
            ]
            concat = torch.cat(all_layers, dim=-1)  # Shape: (batch_size, num_layers * hidden_dim)
            
            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)
            pca_emb = concat @ self.pca_components.T.to(concat.device)  # Shape: (batch_size, pca_dim)
            
            all_embeddings.append(pca_emb.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Encode queries (MTEB interface)"""
        return self.encode(queries, **kwargs)
    
    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)"""
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)
    
    def __del__(self):
        """Close cache on cleanup"""
        self.close_cache()
