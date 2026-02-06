"""
Aggregation methods for combining layer representations.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from .strategies import compute_similarity_weights, normalize_weights, incorporate_layer_quality


class BaseAggregation:
    """Base class for aggregation methods."""

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate layer embeddings.

        Args:
            layer_embeddings: Tensor of shape (num_layers, batch_size, hidden_dim)

        Returns:
            Aggregated embeddings of shape (batch_size, hidden_dim)
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """Get aggregation weights."""
        raise NotImplementedError


class WeightedSumAggregation(BaseAggregation):
    """Weighted sum aggregation based on similarity matrix."""

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        layer_quality: Optional[Dict[int, float]] = None,
        normalize: bool = True
    ):
        """
        Initialize weighted sum aggregation.

        Args:
            similarity_matrix: Layer similarity matrix (num_layers x num_layers)
            layer_quality: Optional quality scores for each layer
            normalize: Whether to normalize weights to sum to 1
        """
        self.similarity_matrix = similarity_matrix
        self.layer_quality = layer_quality
        self.normalize = normalize

        # Compute aggregation weights
        self.weights = compute_similarity_weights(similarity_matrix)

        # Incorporate layer quality if provided
        if layer_quality is not None:
            self.weights = incorporate_layer_quality(self.weights, layer_quality)

        # Normalize if requested
        if normalize:
            self.weights = normalize_weights(self.weights)

        # Convert to torch tensor
        self.weights_tensor = torch.from_numpy(self.weights).float()

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using weighted sum.

        Args:
            layer_embeddings: Shape (num_layers, batch_size, hidden_dim)

        Returns:
            Aggregated embeddings (batch_size, hidden_dim)
        """
        # Move weights to same device as embeddings
        weights = self.weights_tensor.to(layer_embeddings.device)

        # Reshape weights for broadcasting: (num_layers, 1, 1)
        weights = weights.view(-1, 1, 1)

        # Weighted sum: (num_layers, batch_size, hidden_dim) * (num_layers, 1, 1)
        weighted = layer_embeddings * weights

        # Sum over layers
        aggregated = weighted.sum(dim=0)  # (batch_size, hidden_dim)

        return aggregated

    def get_weights(self) -> np.ndarray:
        """Get aggregation weights."""
        return self.weights


class LearnedProjectionAggregation(BaseAggregation, nn.Module):
    """Learned projection-based aggregation."""

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        hidden_dim: int,
        layer_quality: Optional[Dict[int, float]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize learned projection aggregation.

        Args:
            similarity_matrix: Layer similarity matrix
            hidden_dim: Hidden dimension size
            layer_quality: Optional layer quality scores
            dropout: Dropout probability
        """
        super().__init__()
        nn.Module.__init__(self)

        self.similarity_matrix = similarity_matrix
        self.layer_quality = layer_quality
        self.hidden_dim = hidden_dim
        self.num_layers = similarity_matrix.shape[0]

        # Initialize with similarity-based weights
        init_weights = compute_similarity_weights(similarity_matrix)
        if layer_quality is not None:
            init_weights = incorporate_layer_quality(init_weights, layer_quality)
        init_weights = normalize_weights(init_weights)

        # Learnable layer weights
        self.layer_weights = nn.Parameter(torch.from_numpy(init_weights).float())

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_layers, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using learned projection.

        Args:
            layer_embeddings: Shape (num_layers, batch_size, hidden_dim)

        Returns:
            Aggregated embeddings (batch_size, hidden_dim)
        """
        batch_size = layer_embeddings.shape[1]

        # Apply softmax to weights
        weights = torch.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1)

        # Weighted representations
        weighted = layer_embeddings * weights

        # Concatenate all layers
        concatenated = layer_embeddings.permute(1, 0, 2).reshape(batch_size, -1)

        # Project
        aggregated = self.projection(concatenated)

        return aggregated

    def get_weights(self) -> np.ndarray:
        """Get current learned weights."""
        with torch.no_grad():
            weights = torch.softmax(self.layer_weights, dim=0)
            return weights.cpu().numpy()

    def train(
        self,
        layer_embs_train: torch.Tensor,
        labels_train: torch.Tensor,
        layer_embs_val: torch.Tensor,
        labels_val: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10
    ):
        """
        Train the aggregation module.

        Args:
            layer_embs_train: Training layer embeddings
            labels_train: Training labels
            layer_embs_val: Validation layer embeddings
            labels_val: Validation labels
            epochs: Number of epochs
            lr: Learning rate
            patience: Early stopping patience

        Returns:
            List of training losses
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Simple classifier head for training
        num_classes = len(torch.unique(labels_train))
        classifier = nn.Linear(self.hidden_dim, num_classes).to(layer_embs_train.device)

        best_val_loss = float('inf')
        patience_counter = 0
        losses = []

        for epoch in range(epochs):
            # Training
            self.train()
            aggregated = self.aggregate(layer_embs_train)
            logits = classifier(aggregated)
            loss = criterion(logits, labels_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Validation
            self.eval()
            with torch.no_grad():
                val_aggregated = self.aggregate(layer_embs_val)
                val_logits = classifier(val_aggregated)
                val_loss = criterion(val_logits, labels_val)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return losses


class AttentionAggregation(BaseAggregation, nn.Module):
    """Attention-based aggregation."""

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        hidden_dim: int,
        layer_quality: Optional[Dict[int, float]] = None,
        num_heads: int = 8
    ):
        """
        Initialize attention aggregation.

        Args:
            similarity_matrix: Layer similarity matrix
            hidden_dim: Hidden dimension
            layer_quality: Optional layer quality scores
            num_heads: Number of attention heads
        """
        super().__init__()
        nn.Module.__init__(self)

        self.similarity_matrix = similarity_matrix
        self.layer_quality = layer_quality
        self.hidden_dim = hidden_dim
        self.num_layers = similarity_matrix.shape[0]
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=False
        )

        # Layer embeddings (learnable positional embeddings for layers)
        self.layer_embeddings = nn.Parameter(
            torch.randn(self.num_layers, hidden_dim) * 0.02
        )

        # Initialize with similarity bias
        init_weights = compute_similarity_weights(similarity_matrix)
        if layer_quality is not None:
            init_weights = incorporate_layer_quality(init_weights, layer_quality)
        self.register_buffer('similarity_bias', torch.from_numpy(init_weights).float())

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using attention mechanism.

        Args:
            layer_embeddings: Shape (num_layers, batch_size, hidden_dim)

        Returns:
            Aggregated embeddings (batch_size, hidden_dim)
        """
        # Add layer positional embeddings
        layer_pos = self.layer_embeddings.unsqueeze(1)  # (num_layers, 1, hidden_dim)
        layer_embeddings_pos = layer_embeddings + layer_pos

        # Self-attention over layers
        attended, _ = self.attention(
            layer_embeddings_pos,
            layer_embeddings_pos,
            layer_embeddings_pos
        )

        # Weight by similarity bias and average
        weights = torch.softmax(self.similarity_bias, dim=0).view(-1, 1, 1)
        weighted = attended * weights
        aggregated = weighted.sum(dim=0)

        return aggregated

    def get_weights(self) -> np.ndarray:
        """Get attention weights (approximation using similarity bias)."""
        with torch.no_grad():
            weights = torch.softmax(self.similarity_bias, dim=0)
            return weights.cpu().numpy()
