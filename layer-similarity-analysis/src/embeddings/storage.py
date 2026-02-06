"""
Utilities for saving and loading embeddings.
"""

import torch
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def save_embeddings(
    embeddings: Dict[int, torch.Tensor],
    output_dir: Path,
    batch_idx: int = 0
):
    """
    Save embeddings to disk.

    Args:
        embeddings: Dictionary mapping layer_idx -> embeddings tensor
        output_dir: Output directory
        batch_idx: Batch index for naming
    """
    output_dir = Path(output_dir)

    for layer_idx, layer_embeddings in embeddings.items():
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        file_path = layer_dir / f"batch_{batch_idx}.pt"
        torch.save(layer_embeddings, file_path)

    logger.debug(f"Saved embeddings batch {batch_idx} to {output_dir}")


def load_embeddings(
    embeddings_dir: Path,
    num_layers: int
) -> Dict[int, torch.Tensor]:
    """
    Load embeddings from disk.

    Args:
        embeddings_dir: Directory containing embeddings
        num_layers: Number of layers to load

    Returns:
        Dictionary mapping layer_idx -> concatenated embeddings
    """
    embeddings_dir = Path(embeddings_dir)
    all_embeddings = {}

    for layer_idx in range(num_layers + 1):
        layer_dir = embeddings_dir / f"layer_{layer_idx}"

        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        # Load all batches for this layer
        batch_files = sorted(layer_dir.glob("batch_*.pt"), 
                           key=lambda x: int(x.stem.split('_')[1]))

        layer_embeddings = []
        for batch_file in batch_files:
            batch_emb = torch.load(batch_file)
            layer_embeddings.append(batch_emb)

        # Concatenate batches
        all_embeddings[layer_idx] = torch.cat(layer_embeddings, dim=0)

    logger.info(f"Loaded embeddings for {num_layers + 1} layers from {embeddings_dir}")
    return all_embeddings
