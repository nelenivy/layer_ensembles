"""
PCA-based layer aggregation encoders.
Matches notebook implementations for concatPCA methods.
Includes pooler_output support for SimCSE models.
Implements MTEB EncoderProtocol for full compatibility.
"""

import torch
import numpy as np
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, AutoModel

from mteb.models import EncoderProtocol  # ✅ CORRECT IMPORT FROM ORIGINAL


class SelectedLayersPCAEncoder(EncoderProtocol):  # ✅ CORRECT INHERITANCE
    """
    QP-weighted layer selection → Concat with sqrt(weights) → PCA.
    Matches notebook SelectedLayersPCAEncoder (concat+pca+qp).
    Supports SimCSE pooler_output.
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
        pooler_idx: Optional[int] = None
    ):
        """
        Args:
            model_name: HuggingFace model name
            layer_indices: Which layers to use
            layer_weights: Weights for each layer (will apply sqrt!)
            pca_components: PCA components matrix (n_components, concat_dim)
            pca_mean: PCA mean vector (concat_dim,)
            pooling: 'mean' or 'cls'
            batch_size: Batch size for encoding
            device: Device to use
            pooler_idx: Index for pooler_output (e.g., 13 for 12-layer model)
        """
        self.model_name = model_name
        self.layer_indices = layer_indices
        self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)
        self.weights_sqrt = torch.sqrt(self.layer_weights)  # CRITICAL: sqrt!
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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

        # Validate layer indices
        max_idx = self.model.config.num_hidden_layers
        for idx in layer_indices:
            if idx != self.pooler_idx and not (0 <= idx <= max_idx):
                raise ValueError(f"Layer index {idx} out of range or not pooler_idx")

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences using weighted concat → PCA.

        Args:
            sentences: List of sentences to encode
            batch_size: Override default batch size

        Returns:
            embeddings: (N, pca_dim) numpy array
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of (B, T, H)

            # Extract and weight selected layers
            weighted_layers = []
            for idx, layer_idx in enumerate(self.layer_indices):
                if layer_idx == self.pooler_idx:
                    # Use pooler_output
                    if outputs.pooler_output is None:
                        raise RuntimeError(f"pooler_output not available")
                    pooled = outputs.pooler_output  # (B, H)
                else:
                    # Use hidden_states[layer_idx]
                    layer_hidden = hidden_states[layer_idx]  # (B, T, H)

                    if self.pooling == "cls":
                        pooled = layer_hidden[:, 0, :]  # (B, H)
                    else:  # mean
                        mask = inputs['attention_mask'].unsqueeze(-1).float()  # (B, T, 1)
                        pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, H)

                # Apply sqrt(weight) - CRITICAL from notebooks!
                weighted = pooled * self.weights_sqrt[idx].to(pooled.device)
                weighted_layers.append(weighted)

            # Concatenate weighted layers
            concat = torch.cat(weighted_layers, dim=-1)  # (B, concat_dim)

            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)

            pca_emb = concat @ self.pca_components.T.to(concat.device)  # (B, pca_dim)

            all_embeddings.append(pca_emb.cpu().numpy())

        return np.vstack(all_embeddings)

    # MTEB interface compatibility
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)


class ClusterPCAEncoder(EncoderProtocol):  # ✅ CORRECT INHERITANCE
    """
    Hierarchical clustering → Weighted cluster concat → PCA.
    Matches notebook MultiLayerEncoder (concat+pca+cluster).
    Supports SimCSE pooler_output.
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
        pooler_idx: Optional[int] = None
    ):
        """
        Args:
            model_name: HuggingFace model name
            clusters: List of layer clusters, e.g., [[0,1,2], [3,4,5], ...]
            cluster_weights: Weight for each cluster
            pca_components: PCA components matrix
            pca_mean: PCA mean vector
            pooling: 'mean' or 'cls'
            batch_size: Batch size
            device: Device to use
            pooler_idx: Index for pooler_output
        """
        self.model_name = model_name
        self.clusters = clusters
        self.cluster_weights = torch.tensor(cluster_weights, dtype=torch.float32)
        self.weights_sqrt = torch.sqrt(self.cluster_weights)  # CRITICAL: sqrt!
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences using cluster concat → PCA.

        Args:
            sentences: List of sentences
            batch_size: Override batch size

        Returns:
            embeddings: (N, pca_dim) numpy array
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of (B, T, H)

            # Process each cluster
            cluster_vecs = []
            for cluster_idx, cluster_layers in enumerate(self.clusters):
                # Average layers within cluster
                layer_embeddings = []
                for layer_idx in cluster_layers:
                    if layer_idx == self.pooler_idx:
                        # Use pooler_output
                        if outputs.pooler_output is None:
                            raise RuntimeError(f"pooler_output not available")
                        pooled = outputs.pooler_output
                    else:
                        # Use hidden_states[layer_idx]
                        layer_hidden = hidden_states[layer_idx]  # (B, T, H)

                        if self.pooling == "cls":
                            pooled = layer_hidden[:, 0, :]
                        else:  # mean
                            mask = inputs['attention_mask'].unsqueeze(-1).float()
                            pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                    layer_embeddings.append(pooled)

                # Average within cluster
                cluster_avg = torch.stack(layer_embeddings, dim=0).mean(dim=0)  # (B, H)

                # Apply sqrt(weight)
                weighted = cluster_avg * self.weights_sqrt[cluster_idx].to(cluster_avg.device)
                cluster_vecs.append(weighted)

            # Concatenate clusters
            concat = torch.cat(cluster_vecs, dim=-1)  # (B, num_clusters * H)

            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)

            pca_emb = concat @ self.pca_components.T.to(concat.device)  # (B, pca_dim)

            all_embeddings.append(pca_emb.cpu().numpy())

        return np.vstack(all_embeddings)

    # MTEB interface compatibility
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)


class AllLayersPCAEncoder(EncoderProtocol):  # ✅ CORRECT INHERITANCE
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
        device: str = "cuda"
    ):
        """
        Args:
            model_name: HuggingFace model name
            pca_components: PCA components matrix
            pca_mean: PCA mean vector
            pooling: 'mean' or 'cls'
            batch_size: Batch size
            device: Device
        """
        self.model_name = model_name
        self.pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32) if pca_mean is not None else None
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()

        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences by concatenating all layers → PCA.

        Args:
            sentences: List of sentences
            batch_size: Override batch size

        Returns:
            embeddings: (N, pca_dim) numpy array
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of (B, T, H)

            # Pool all layers
            pooled_layers = []
            for layer_hidden in hidden_states:
                if self.pooling == "cls":
                    pooled = layer_hidden[:, 0, :]
                else:  # mean
                    mask = inputs['attention_mask'].unsqueeze(-1).float()
                    pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled_layers.append(pooled)

            # Concatenate all layers
            concat = torch.cat(pooled_layers, dim=-1)  # (B, num_layers * H)

            # Apply PCA
            if self.pca_mean is not None:
                concat = concat - self.pca_mean.to(concat.device)

            pca_emb = concat @ self.pca_components.T.to(concat.device)  # (B, pca_dim)

            all_embeddings.append(pca_emb.cpu().numpy())

        return np.vstack(all_embeddings)

    # MTEB interface compatibility
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]]],
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                text = [doc.get(key) for key in ("text", "title", "content") if doc.get(key)]
                if text:
                    sentences.append(str(text[0]))
            return self.encode(sentences, **kwargs)
        return self.encode(corpus, **kwargs)
