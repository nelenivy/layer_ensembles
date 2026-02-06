"""
Data loading utilities for C4 and MTEB datasets.
"""

from datasets import load_dataset as hf_load_dataset
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    num_samples: int = 10000,
    split: str = 'train',
    text_column: Optional[str] = None,
    seed: int = 42
) -> List[str]:
    """
    Load dataset and extract text samples.

    Args:
        dataset_name: Dataset name ('c4' or MTEB task name)
        num_samples: Number of samples to load
        split: Dataset split
        text_column: Name of text column (auto-detected if None)
        seed: Random seed for sampling

    Returns:
        List of text strings
    """
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Samples: {num_samples}, Split: {split}")

    if dataset_name.lower() == 'c4':
        return _load_c4(num_samples, split, seed)
    else:
        return _load_mteb(dataset_name, num_samples, split, text_column, seed)


def _load_c4(num_samples: int, split: str, seed: int) -> List[str]:
    """Load C4 dataset."""
    logger.info("Loading C4 dataset...")

    dataset = hf_load_dataset(
        'allenai/c4',
        'en',
        split=split,
        streaming=True
    )

    # Sample from stream
    texts = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        texts.append(example['text'])

    logger.info(f"Loaded {len(texts)} samples from C4")
    return texts


def _load_mteb(
    dataset_name: str,
    num_samples: int,
    split: str,
    text_column: Optional[str],
    seed: int
) -> List[str]:
    """Load MTEB dataset."""
    try:
        from mteb import MTEB
    except ImportError:
        raise ImportError("MTEB not installed. Install with: pip install mteb")

    logger.info(f"Loading MTEB task: {dataset_name}")

    # Load MTEB task
    tasks = MTEB(tasks=[dataset_name])
    task = tasks.tasks[0]

    # Load dataset
    task.load_data()

    # Get the appropriate split
    if hasattr(task, 'dataset') and split in task.dataset:
        dataset = task.dataset[split]
    else:
        raise ValueError(f"Split '{split}' not found in {dataset_name}")

    # Auto-detect text column if not specified
    if text_column is None:
        # Common column names in MTEB
        possible_columns = ['text', 'sentence', 'query', 'sentence1', 'sent1']
        for col in possible_columns:
            if col in dataset.column_names:
                text_column = col
                break

        if text_column is None:
            # Use first text column
            text_column = dataset.column_names[0]

        logger.info(f"Auto-detected text column: {text_column}")

    # Sample texts
    if len(dataset) <= num_samples:
        texts = dataset[text_column]
    else:
        # Random sample
        indices = list(range(len(dataset)))
        import random
        random.seed(seed)
        random.shuffle(indices)
        sampled_indices = indices[:num_samples]
        texts = [dataset[i][text_column] for i in sampled_indices]

    logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
    return texts
