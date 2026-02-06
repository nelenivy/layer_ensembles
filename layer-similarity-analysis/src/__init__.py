"""
Layer Similarity Analysis Framework

A modular framework for analyzing representational similarity across transformer layers.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models.pooling import POOLING_STRATEGIES
from src.embeddings.extractor import EmbeddingExtractor
from src.metrics.calculator import SimilarityCalculator
from src.data.loaders import load_dataset

__all__ = [
    'POOLING_STRATEGIES',
    'EmbeddingExtractor',
    'SimilarityCalculator',
    'load_dataset',
]
