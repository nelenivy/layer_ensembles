"""
Unit tests for encoder modules.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import tempfile
import pickle
import numpy as np
from src.models.encoders import SimpleEncoder, LayerEncoder


def test_simple_encoder_initialization():
    """Test SimpleEncoder initialization."""
    encoder = SimpleEncoder("bert-base-uncased", pooling="mean", layer_idx=6)
    assert encoder.model_name == "bert-base-uncased"
    assert encoder.pooling == "mean"
    assert encoder.layer_idx == 6


def test_simple_encoder_encoding():
    """Test SimpleEncoder text encoding."""
    encoder = SimpleEncoder("bert-base-uncased", pooling="mean", layer_idx=6, batch_size=2)

    texts = ["Hello world", "This is a test"]
    embeddings = encoder.encode(texts)

    assert embeddings.shape[0] == 2  # batch size
    assert embeddings.shape[1] == 768  # BERT dimension
    assert isinstance(embeddings, torch.Tensor)


def test_simple_encoder_pooling_methods():
    """Test different pooling methods."""
    texts = ["Test sentence"]

    for pooling in ["mean", "cls", "last_token"]:
        encoder = SimpleEncoder("bert-base-uncased", pooling=pooling, layer_idx=0)
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (1, 768)


def test_layer_encoder_initialization():
    """Test LayerEncoder initialization."""
    encoder = LayerEncoder("bert-base-uncased", pooling="mean")

    # Check that all layers are accessible
    num_layers = encoder.model.config.num_hidden_layers + 1
    assert len(encoder.get_all_layer_indices()) == num_layers


def test_layer_encoder_single_layer():
    """Test LayerEncoder with single layer."""
    encoder = LayerEncoder("bert-base-uncased", pooling="mean")

    texts = ["Hello world"]
    embeddings = encoder.encode(texts, layer_idx=6)

    assert embeddings.shape == (1, 768)


def test_layer_encoder_all_layers():
    """Test LayerEncoder with all layers."""
    encoder = LayerEncoder("bert-base-uncased", pooling="mean")

    texts = ["Hello world", "Test"]
    all_embeddings = encoder.encode_all_layers(texts)

    num_layers = encoder.model.config.num_hidden_layers + 1
    assert all_embeddings.shape[0] == num_layers
    assert all_embeddings.shape[1] == 2  # batch size
    assert all_embeddings.shape[2] == 768  # dimension


def test_encoder_batch_processing():
    """Test batch processing with larger input."""
    encoder = SimpleEncoder("bert-base-uncased", pooling="mean", batch_size=4)

    texts = [f"Test sentence {i}" for i in range(10)]
    embeddings = encoder.encode(texts)

    assert embeddings.shape[0] == 10
    assert embeddings.shape[1] == 768


@pytest.mark.parametrize("layer_idx", [0, 6, 12])
def test_encoder_layer_selection(layer_idx):
    """Test encoding from different layers."""
    encoder = SimpleEncoder("bert-base-uncased", layer_idx=layer_idx)
    embeddings = encoder.encode(["Test"])

    assert embeddings.shape == (1, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
