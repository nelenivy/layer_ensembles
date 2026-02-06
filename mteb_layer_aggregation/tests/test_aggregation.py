"""
Unit tests for aggregation methods.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
import tempfile
import pickle
from src.aggregation.methods import WeightedSumAggregation, LearnedProjectionAggregation
from src.aggregation.strategies import compute_similarity_weights, normalize_weights


def create_dummy_similarity_matrix(n_layers=13, random_state=42):
    """Create a dummy similarity matrix for testing."""
    np.random.seed(random_state)
    # Create symmetric matrix with high values on diagonal
    matrix = np.random.rand(n_layers, n_layers) * 0.5
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return matrix


def test_compute_similarity_weights():
    """Test similarity weight computation."""
    sim_matrix = create_dummy_similarity_matrix(13)
    weights = compute_similarity_weights(sim_matrix)

    assert weights.shape == (13,)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)


def test_normalize_weights():
    """Test weight normalization."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = normalize_weights(weights)

    assert normalized.shape == weights.shape
    assert np.isclose(normalized.sum(), 1.0)
    assert np.all(normalized >= 0)


def test_weighted_sum_aggregation():
    """Test WeightedSumAggregation."""
    # Create dummy data
    n_layers, batch_size, hidden_dim = 13, 4, 768
    layer_embeddings = torch.randn(n_layers, batch_size, hidden_dim)
    sim_matrix = create_dummy_similarity_matrix(n_layers)

    # Create aggregation method
    aggregator = WeightedSumAggregation(sim_matrix)

    # Aggregate
    aggregated = aggregator.aggregate(layer_embeddings)

    assert aggregated.shape == (batch_size, hidden_dim)
    assert isinstance(aggregated, torch.Tensor)


def test_weighted_sum_with_layer_quality():
    """Test WeightedSumAggregation with layer quality."""
    n_layers = 13
    layer_embeddings = torch.randn(n_layers, 4, 768)
    sim_matrix = create_dummy_similarity_matrix(n_layers)

    # Create quality scores (higher for middle layers)
    quality = {i: 0.5 + 0.1 * (6 - abs(i - 6)) for i in range(n_layers)}

    aggregator = WeightedSumAggregation(sim_matrix, layer_quality=quality)
    aggregated = aggregator.aggregate(layer_embeddings)

    assert aggregated.shape == (4, 768)


def test_learned_projection_aggregation():
    """Test LearnedProjectionAggregation."""
    n_layers, batch_size, hidden_dim = 13, 4, 768
    layer_embeddings = torch.randn(n_layers, batch_size, hidden_dim)
    sim_matrix = create_dummy_similarity_matrix(n_layers)

    aggregator = LearnedProjectionAggregation(
        sim_matrix=sim_matrix,
        hidden_dim=hidden_dim
    )

    aggregated = aggregator.aggregate(layer_embeddings)

    assert aggregated.shape == (batch_size, hidden_dim)


def test_learned_projection_training():
    """Test training learned projection."""
    n_layers, hidden_dim = 13, 768
    sim_matrix = create_dummy_similarity_matrix(n_layers)

    aggregator = LearnedProjectionAggregation(sim_matrix, hidden_dim)

    # Create dummy training data
    layer_embs_train = torch.randn(n_layers, 100, hidden_dim)
    labels_train = torch.randint(0, 5, (100,))

    layer_embs_val = torch.randn(n_layers, 20, hidden_dim)
    labels_val = torch.randint(0, 5, (20,))

    # Train for a few epochs
    losses = aggregator.train(
        layer_embs_train,
        labels_train,
        layer_embs_val,
        labels_val,
        epochs=5
    )

    assert len(losses) > 0
    assert all(isinstance(l, float) for l in losses)


def test_aggregation_with_different_batch_sizes():
    """Test aggregation with various batch sizes."""
    sim_matrix = create_dummy_similarity_matrix(13)
    aggregator = WeightedSumAggregation(sim_matrix)

    for batch_size in [1, 4, 16, 32]:
        layer_embeddings = torch.randn(13, batch_size, 768)
        aggregated = aggregator.aggregate(layer_embeddings)
        assert aggregated.shape == (batch_size, 768)


@pytest.mark.parametrize("n_layers", [7, 13, 25])
def test_aggregation_different_layer_counts(n_layers):
    """Test aggregation with different numbers of layers."""
    sim_matrix = create_dummy_similarity_matrix(n_layers)
    aggregator = WeightedSumAggregation(sim_matrix)

    layer_embeddings = torch.randn(n_layers, 4, 768)
    aggregated = aggregator.aggregate(layer_embeddings)

    assert aggregated.shape == (4, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
