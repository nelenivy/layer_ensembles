"""
Data loading utilities for layer aggregation experiments.
"""
import pickle
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def load_similarity_matrix(filepath: str) -> np.ndarray:
    """Load similarity matrix from pickle file."""
    with open(filepath, 'rb') as f:
        sim_matrix = pickle.load(f)
    return np.asarray(sim_matrix, dtype=np.float32)


def save_results(data: Dict, filepath: str):
    """Save results to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(filepath: str) -> Dict:
    """Load results from pickle file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def load_c4_dataset(
    n_samples: int = 10000,
    seed: int = 42,
    streaming: bool = True
) -> List[str]:
    """
    Load C4 dataset for PCA training or layer quality evaluation.

    Args:
        n_samples: Number of samples to load
        seed: Random seed
        streaming: Use streaming mode

    Returns:
        List of text samples
    """
    ds = load_dataset("allenai/c4", "en", split="train", streaming=streaming)
    texts = [ex["text"] for ex in ds.shuffle(buffer_size=n_samples, seed=seed).take(n_samples)]
    return texts


def load_classification_dataset(
    dataset_name: str,
    test_size: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    """
    Load and split classification dataset for layer evaluation.

    Args:
        dataset_name: MTEB dataset name (e.g., "mteb/banking77")
        test_size: Test split ratio
        seed: Random seed

    Returns:
        DatasetDict with train/dev/test splits
    """
    # Load dataset
    ds = load_dataset(dataset_name)

    # Create train/dev split if not exists
    if "train" in ds and "test" in ds and "dev" not in ds:
        train_val = ds["train"].train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({
            "train": train_val["train"],
            "dev": train_val["test"],
            "test": ds["test"]
        })

    return ds


def load_retrieval_dataset(dataset_name: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load retrieval dataset with queries, corpus, and relevance judgments.

    Args:
        dataset_name: MTEB dataset name

    Returns:
        Tuple of (queries, corpus, relevance_judgments)
    """
    # Load components
    ds_relevance = load_dataset(dataset_name, split="dev")
    queries = load_dataset(dataset_name, name="queries", split="queries")
    corpus = load_dataset(dataset_name, name="corpus", split="corpus")

    return list(queries), list(corpus), list(ds_relevance)


def prepare_layer_quality_data(
    dataset_name: str,
    text_column: str = "text",
    label_column: str = "label",
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[int]]:
    """
    Prepare data for layer quality evaluation.

    Args:
        dataset_name: Dataset name
        text_column: Name of text column
        label_column: Name of label column
        max_samples: Maximum samples to use

    Returns:
        Tuple of (texts, labels)
    """
    ds = load_dataset(dataset_name)

    # Get appropriate split
    split = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split]

    texts = data[text_column]
    labels = data[label_column]

    if max_samples is not None:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    return texts, labels


def find_similarity_matrices(
    base_dir: str,
    model_name: str,
    pooling: str,
    dataset: str
) -> Dict[str, str]:
    """
    Find similarity matrix files from grid search results.

    Args:
        base_dir: Base directory containing grid search results
        model_name: Model name (e.g., "bert-base-uncased")
        pooling: Pooling method (e.g., "mean")
        dataset: Dataset name (e.g., "c4")

    Returns:
        Dictionary mapping metric names to file paths
    """
    base_path = Path(base_dir)

    # Sanitize name for directory format
    model_dir = model_name.replace("/", "_").replace("-", "_")
    exp_dir = base_path / f"{model_dir}_{pooling}_{dataset}"

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Find all similarity matrices
    sim_matrices = {}
    for metric in ["CKA", "RSA", "correlation", "cosine"]:
        sim_file = exp_dir / f"{metric}.pkl"
        if sim_file.exists():
            sim_matrices[metric] = str(sim_file)

    return sim_matrices


def load_layer_quality_results(
    filepath: str
) -> Dict[int, float]:
    """
    Load layer quality evaluation results.

    Args:
        filepath: Path to results pickle file

    Returns:
        Dictionary mapping layer indices to quality scores
    """
    results = load_results(filepath)

    # Extract quality scores (assumes results have 'accuracy' or similar metric)
    layer_quality = {}
    for layer_idx, metrics in results.items():
        if isinstance(metrics, dict):
            score = metrics.get("accuracy", metrics.get("score", 0.0))
        else:
            score = float(metrics)
        layer_quality[layer_idx] = score

    return layer_quality
