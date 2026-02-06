#!/usr/bin/env python
"""
Script for visualizing similarity matrices.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_similarity_matrix(
    matrix: np.ndarray,
    title: str = "Layer Similarity Matrix",
    output_path: str = None,
    cmap: str = "viridis"
):
    """Plot similarity matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=False,
        square=True,
        cbar_kws={"label": "Similarity"},
        ax=ax
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize similarity matrices")
    parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        help="Path to similarity matrix pickle file"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Layer Similarity Matrix",
        help="Plot title"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap name"
    )

    args = parser.parse_args()

    # Load similarity matrix
    print(f"Loading similarity matrix from: {args.matrix}")
    with open(args.matrix, 'rb') as f:
        sim_matrix = pickle.load(f)

    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)

    print(f"Matrix shape: {sim_matrix.shape}")
    print(f"Value range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # Plot
    plot_similarity_matrix(
        sim_matrix,
        title=args.title,
        output_path=args.output,
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()
