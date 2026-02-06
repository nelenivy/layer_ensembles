"""
Similarity metric calculation with Monte Carlo sampling.

Implements:
- CKA (Centered Kernel Alignment)
- RSA (Representational Similarity Analysis)
- Jaccard Similarity
- Distance Correlation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# Available metrics
AVAILABLE_METRICS = ['CKA', 'RSA', 'JaccardSimilarity', 'DistanceCorrelation']


class SimilarityCalculator:
    """
    Calculate similarity between layer representations.

    Uses Monte Carlo sampling for efficiency:
    - Sample subsets multiple times
    - Compute similarity on each sample
    - Average results and compute std

    Args:
        sample_size: Number of samples per trial
        num_trials: Number of Monte Carlo trials
        seed: Random seed
    """

    def __init__(
        self,
        sample_size: int = 3000,
        num_trials: int = 5,
        seed: int = 42
    ):
        self.sample_size = sample_size
        self.num_trials = num_trials
        self.seed = seed

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

    def calculate_all(
        self,
        embeddings: Dict[int, torch.Tensor],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate all similarity metrics.

        Args:
            embeddings: Dictionary mapping layer_idx -> embeddings [N, D]
            metrics: List of metrics to calculate (default: all)

        Returns:
            Dictionary mapping metric_name -> {'mean': matrix, 'std': matrix}
        """
        if metrics is None:
            metrics = AVAILABLE_METRICS

        num_layers = len(embeddings)
        results = {}

        for metric_name in metrics:
            logger.info(f"Calculating {metric_name}...")

            mean_matrix, std_matrix = self._calculate_metric(
                embeddings,
                metric_name
            )

            results[metric_name] = {
                'mean': mean_matrix,
                'std': std_matrix
            }

        return results

    def _calculate_metric(
        self,
        embeddings: Dict[int, torch.Tensor],
        metric_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate similarity matrix for a metric with Monte Carlo sampling.

        Returns:
            (mean_matrix, std_matrix) of shape [num_layers, num_layers]
        """
        num_layers = len(embeddings)
        num_samples = embeddings[0].shape[0]

        # Storage for trials
        all_matrices = []

        for trial in range(self.num_trials):
            # Sample indices
            if num_samples <= self.sample_size:
                indices = list(range(num_samples))
            else:
                indices = np.random.choice(num_samples, self.sample_size, replace=False)

            # Sample embeddings
            sampled_emb = {
                layer: embeddings[layer][indices].numpy()
                for layer in range(num_layers)
            }

            # Compute similarity matrix
            similarity_matrix = np.zeros((num_layers, num_layers))

            for i in range(num_layers):
                for j in range(i, num_layers):
                    sim = self._compute_similarity(
                        sampled_emb[i],
                        sampled_emb[j],
                        metric_name
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

            all_matrices.append(similarity_matrix)

        # Compute mean and std
        all_matrices = np.array(all_matrices)  # [num_trials, num_layers, num_layers]
        mean_matrix = np.mean(all_matrices, axis=0)
        std_matrix = np.std(all_matrices, axis=0)

        return mean_matrix, std_matrix

    def _compute_similarity(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        metric_name: str
    ) -> float:
        """
        Compute similarity between two representation matrices.

        Args:
            X: [N, D1] array
            Y: [N, D2] array
            metric_name: Name of metric

        Returns:
            Similarity score
        """
        if metric_name == 'CKA':
            return self._cka(X, Y)
        elif metric_name == 'RSA':
            return self._rsa(X, Y)
        elif metric_name == 'JaccardSimilarity':
            return self._jaccard_similarity(X, Y)
        elif metric_name == 'DistanceCorrelation':
            return self._distance_correlation(X, Y)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Centered Kernel Alignment."""
        # Linear CKA
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Gram matrices
        K = X @ X.T
        L = Y @ Y.T

        # CKA
        hsic = np.trace(K @ L)
        normalization = np.sqrt(np.trace(K @ K) * np.trace(L @ L))

        if normalization == 0:
            return 0.0

        return hsic / normalization

    def _rsa(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Representational Similarity Analysis."""
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import spearmanr

        # Compute RDMs (Representational Dissimilarity Matrices)
        rdm_x = squareform(pdist(X, metric='correlation'))
        rdm_y = squareform(pdist(Y, metric='correlation'))

        # Flatten and compute Spearman correlation
        rdm_x_flat = rdm_x[np.triu_indices_from(rdm_x, k=1)]
        rdm_y_flat = rdm_y[np.triu_indices_from(rdm_y, k=1)]

        correlation, _ = spearmanr(rdm_x_flat, rdm_y_flat)

        return correlation

    def _jaccard_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Jaccard Similarity of top-k activated features."""
        k = min(100, X.shape[1] // 10)  # Top 10% or max 100

        # Get top-k indices for each sample
        top_k_x = np.argsort(X, axis=1)[:, -k:]
        top_k_y = np.argsort(Y, axis=1)[:, -k:]

        # Compute average Jaccard across samples
        jaccards = []
        for i in range(len(X)):
            set_x = set(top_k_x[i])
            set_y = set(top_k_y[i])

            intersection = len(set_x & set_y)
            union = len(set_x | set_y)

            if union > 0:
                jaccards.append(intersection / union)

        return np.mean(jaccards) if jaccards else 0.0

    def _distance_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Distance Correlation."""
        from scipy.spatial.distance import pdist, squareform

        # Compute distance matrices
        dist_x = squareform(pdist(X))
        dist_y = squareform(pdist(Y))

        # Center distance matrices
        n = len(dist_x)
        A = dist_x - dist_x.mean(axis=0, keepdims=True) - dist_x.mean(axis=1, keepdims=True) + dist_x.mean()
        B = dist_y - dist_y.mean(axis=0, keepdims=True) - dist_y.mean(axis=1, keepdims=True) + dist_y.mean()

        # Distance covariance
        dcov_xy = np.sqrt((A * B).sum() / (n ** 2))
        dcov_xx = np.sqrt((A * A).sum() / (n ** 2))
        dcov_yy = np.sqrt((B * B).sum() / (n ** 2))

        # Distance correlation
        if dcov_xx == 0 or dcov_yy == 0:
            return 0.0

        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
