"""
Visualization utilities for similarity matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    title: str = "Layer Similarity",
    layer_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f'
):
    """
    Plot similarity matrix as heatmap.

    Args:
        similarity_matrix: Similarity matrix [N, N]
        title: Plot title
        layer_names: Layer labels (default: Layer 0, Layer 1, ...)
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        annot: Show annotations
        fmt: Annotation format
    """
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(len(similarity_matrix))]

    plt.figure(figsize=figsize)

    sns.heatmap(
        similarity_matrix,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        xticklabels=layer_names,
        yticklabels=layer_names,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Similarity'}
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()


def plot_dendrogram(
    similarity_matrix: np.ndarray,
    title: str = "Layer Clustering",
    layer_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    method: str = 'average'
):
    """
    Plot hierarchical clustering dendrogram.

    Args:
        similarity_matrix: Similarity matrix [N, N]
        title: Plot title
        layer_names: Layer labels
        save_path: Path to save figure
        figsize: Figure size
        method: Linkage method ('average', 'single', 'complete', 'ward')
    """
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(similarity_matrix))]

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix

    # Hierarchical clustering
    condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    linkage_matrix = linkage(condensed_dist, method=method)

    plt.figure(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=layer_names,
        leaf_font_size=10
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Distance (1 - Similarity)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dendrogram to {save_path}")

    plt.show()


def plot_layer_progression(
    similarity_matrix: np.ndarray,
    reference_layer: int = 0,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot similarity of all layers to a reference layer.

    Args:
        similarity_matrix: Similarity matrix [N, N]
        reference_layer: Reference layer index
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if title is None:
        title = f"Similarity to Layer {reference_layer}"

    similarities = similarity_matrix[reference_layer, :]
    layers = np.arange(len(similarities))

    plt.figure(figsize=figsize)
    plt.plot(layers, similarities, marker='o', linewidth=2, markersize=8)
    plt.axvline(reference_layer, color='red', linestyle='--', alpha=0.5, label=f'Layer {reference_layer}')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Similarity', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved progression plot to {save_path}")

    plt.show()
