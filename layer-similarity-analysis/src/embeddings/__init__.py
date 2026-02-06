"""Embedding extraction and storage."""

from src.embeddings.extractor import EmbeddingExtractor
from src.embeddings.storage import save_embeddings, load_embeddings

__all__ = [
    'EmbeddingExtractor',
    'save_embeddings',
    'load_embeddings',
]
