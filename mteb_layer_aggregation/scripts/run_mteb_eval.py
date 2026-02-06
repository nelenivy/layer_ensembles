#!/usr/bin/env python3
"""
MTEB Evaluation Script with Layer Aggregation Methods.
Supports weighted sum, greedy, cluster, and PCA-based aggregation methods.

Features:
- Task type filtering (Classification, Retrieval, STS, etc.)
- Automatic filtering of non-text (image/multimodal) tasks
- Incremental result saving after each task
- Filter by dataset size (number of samples)
- Overwrite results control
"""

import os
import sys
import pickle
import logging
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import mteb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.aggregated_encoder import AggregatedEncoder
from src.models.pca_encoders import (
    SelectedLayersPCAEncoder,
    ClusterPCAEncoder,
    AllLayersPCAEncoder
)
from src.aggregation.strategies import (
    compute_similarity_weights,
    compute_greedy_weights,
    compute_greedy_weights_v2,
    compute_layer_clusters,
    compute_cluster_weights_for_pca,
    normalize_weights
)
from scripts.train_pca import (
    train_pca_for_selected_layers,
    train_pca_for_clusters,
    train_pca_all_layers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Hardcoded list of image/multimodal tasks to exclude
IMAGE_TASK_NAMES = [
    "BirdsnapZeroShot", "Caltech101ZeroShot", "CIFAR100ZeroShot",
    "Country211ZeroShot", "DTDZeroShot", "EuroSATZeroShot",
    "FER2013ZeroShot", "FGVCAircraftZeroShot", "Food101ZeroShot",
    "Flowers102ZeroShot", "GTSRBZeroShot", "Imagenet1kZeroShot",
    "OxfordPetsZeroShot", "PatchCamelyonZeroShot", "RESISC45ZeroShot",
    "StanfordCarsZeroShot", "STL10ZeroShot", "SUN397ZeroShot", "UCF101ZeroShot",
    "CIFAR10ZeroShot", "MNISTZeroShot",
    "CIFAR10Clustering", "CIFAR100Clustering", "ImageNetDog15Clustering",
    "TinyImageNetClustering",
    "BLINKIT2IRetrieval", "BLINKIT2TRetrieval", "COCO2017Retrieval",
    "Flickr30kRetrieval", "MIRACLRetrieval", "MIRACLVisionRetrieval",
    "VisualNewsRetrieval", "WebQARetrieval", "Wiki-SS-NQRetrieval",
    "ChartQA", "DocVQA", "InfographicVQA", "OCR-VQA", "TextVQA",
    "SUGARCREPEAddition", "SUGARCREPEReplacement", "SUGARCREPESwap",
    "VOC2007Classification", "ImageNet", "Places365",
]


def filter_text_only_tasks(tasks: List[Any]) -> List[Any]:
    """Filter to keep ONLY text-only tasks."""
    text_only_tasks = []
    filtered_out = []

    for task in tasks:
        task_name = task.metadata.name

        if task_name in IMAGE_TASK_NAMES:
            filtered_out.append(f"{task_name} (known image task)")
            continue

        modalities = getattr(task.metadata, 'modalities', None)
        if modalities:
            if modalities == ["text"] or modalities == ("text",):
                text_only_tasks.append(task)
            else:
                filtered_out.append(f"{task_name} (modalities: {modalities})")
        else:
            text_only_tasks.append(task)

    if filtered_out:
        logger.info(f"Filtered out {len(filtered_out)} image/multimodal tasks")
        for task_info in filtered_out[:15]:
            logger.info(f"  - {task_info}")
        if len(filtered_out) > 15:
            logger.info(f"  ... and {len(filtered_out) - 15} more")

    logger.info(f"Kept {len(text_only_tasks)} text-only tasks for evaluation")
    return text_only_tasks


def filter_by_task_types(tasks: List[Any], task_types: List[str]) -> List[Any]:
    """Filter tasks by task type."""
    if not task_types:
        return tasks

    filtered_tasks = []
    for task in tasks:
        task_type = getattr(task.metadata, 'type', None)
        if task_type and task_type in task_types:
            filtered_tasks.append(task)

    logger.info(f"Filtered to {len(filtered_tasks)} tasks of types: {task_types}")
    return filtered_tasks


def get_task_sample_count(task) -> int:
    """
    Get the number of samples in a task.

    Returns:
        Number of samples, or -1 if cannot determine
    """
    try:
        # Try to load the dataset and count samples
        task.load_data()

        total_samples = 0

        # For classification/clustering tasks
        if hasattr(task, 'dataset') and task.dataset is not None:
            for split_name, split_data in task.dataset.items():
                if split_data is not None:
                    total_samples += len(split_data)

        # For retrieval tasks (corpus + queries)
        if hasattr(task, 'corpus') and task.corpus is not None:
            for split_name, split_data in task.corpus.items():
                if split_data is not None:
                    total_samples += len(split_data)

        if hasattr(task, 'queries') and task.queries is not None:
            for split_name, split_data in task.queries.items():
                if split_data is not None:
                    total_samples += len(split_data)

        return total_samples if total_samples > 0 else -1

    except Exception as e:
        logger.debug(f"Could not determine sample count for {task.metadata.name}: {e}")
        return -1


def filter_by_sample_count(tasks: List[Any], max_samples: Optional[int]) -> List[Any]:
    """
    Filter tasks by maximum number of samples.

    Args:
        tasks: List of MTEB tasks
        max_samples: Maximum number of samples (None = no filtering)

    Returns:
        Filtered list of tasks
    """
    if max_samples is None:
        return tasks

    logger.info(f"\nFiltering tasks with <= {max_samples} samples...")

    filtered_tasks = []
    filtered_out = []

    for task in tasks:
        task_name = task.metadata.name
        sample_count = get_task_sample_count(task)

        if sample_count == -1:
            # Cannot determine size, keep it
            logger.warning(f"Cannot determine size for {task_name}, keeping it")
            filtered_tasks.append(task)
        elif sample_count <= max_samples:
            logger.info(f"✓ {task_name}: {sample_count:,} samples")
            filtered_tasks.append(task)
        else:
            logger.info(f"✗ {task_name}: {sample_count:,} samples (too large)")
            filtered_out.append(f"{task_name} ({sample_count:,} samples)")

    logger.info(f"\nFiltered to {len(filtered_tasks)} tasks with <= {max_samples:,} samples")
    if filtered_out:
        logger.info(f"Excluded {len(filtered_out)} tasks (too large)")

    return filtered_tasks


def load_similarity_matrix(path: str) -> np.ndarray:
    """Load similarity matrix from pickle file."""
    logger.info(f"Loading similarity matrix from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'matrix' in data:
        matrix = data['matrix']
    else:
        matrix = data

    matrix = np.asarray(matrix, dtype=np.float32)
    logger.info(f"Loaded similarity matrix with shape {matrix.shape}")
    return matrix


def compute_layer_quality(similarity_matrix: np.ndarray, method: str = "diagonal") -> np.ndarray:
    """Compute quality score for each layer."""
    if method == "diagonal":
        quality = np.diag(similarity_matrix)
    elif method == "mean":
        quality = similarity_matrix.mean(axis=1)
    else:
        raise ValueError(f"Unknown quality method: {method}")

    quality = np.asarray(quality, dtype=np.float32)
    return quality


def compute_method_weights(
    method: str,
    similarity_matrix: np.ndarray,
    layer_quality: np.ndarray,
    lmbd: float,
    num_clusters: int = 4
) -> np.ndarray:
    """Compute aggregation weights based on method."""
    if method == "weighted":
        weights = compute_similarity_weights(similarity_matrix, layer_quality, lmbd)
        weights = normalize_weights(weights, threshold=0.001)
    elif method == "greedy":
        weights = compute_greedy_weights(similarity_matrix, layer_quality, lmbd)
        weights = normalize_weights(weights, threshold=0.001)
    elif method == "greedyv2":
        weights = compute_greedy_weights_v2(similarity_matrix, layer_quality, lmbd)
        weights = normalize_weights(weights, threshold=0.001)
    elif method == "cluster":
        clusters = compute_layer_clusters(similarity_matrix, num_clusters)
        num_layers = similarity_matrix.shape[0]
        weights = np.zeros(num_layers, dtype=np.float32)
        for cluster in clusters:
            cluster_quality = np.mean([layer_quality[i] for i in cluster])
            for layer_idx in cluster:
                weights[layer_idx] = cluster_quality
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def create_aggregated_encoder(
    model_name: str,
    similarity_matrix: np.ndarray,
    method: str,
    layer_quality: np.ndarray,
    lmbd: float,
    batch_size: int = 32,
    pca_cache_dir: str = "./pca_cache",
    pooling: str = "mean",
    num_clusters: int = 4
):
    """Create encoder with method-specific weights or PCA."""

    if method in ["weighted", "greedy", "greedyv2", "cluster"]:
        weights = compute_method_weights(method, similarity_matrix, layer_quality, lmbd, num_clusters)

        encoder = AggregatedEncoder(
            model_name=model_name,
            similarity_matrix=similarity_matrix,
            pooling=pooling,
            batch_size=batch_size,
            aggregation_weights=weights
        )

    elif method == "concat+pca+qp":
        weights = compute_similarity_weights(similarity_matrix, layer_quality, lmbd)
        weights = normalize_weights(weights, threshold=0.001)

        layer_indices = np.where(weights > 0.001)[0].tolist()
        layer_weights = weights[layer_indices]

        os.makedirs(pca_cache_dir, exist_ok=True)
        model_name_safe = model_name.replace("/", "_")
        pca_path = f"{pca_cache_dir}/{model_name_safe}_qp_lmbd{lmbd}.pkl"

        if not os.path.exists(pca_path):
            logger.info(f"Training PCA for concat+pca+qp (lambda={lmbd})...")
            pca_result = train_pca_for_selected_layers(
                model_name=model_name,
                layer_indices=layer_indices,
                layer_weights=layer_weights,
                output_path=pca_path,
                pooling=pooling
            )
        else:
            logger.info(f"Loading PCA from {pca_path}")
            with open(pca_path, 'rb') as f:
                pca_result = pickle.load(f)

        encoder = SelectedLayersPCAEncoder(
            model_name=model_name,
            layer_indices=layer_indices,
            layer_weights=layer_weights,
            pca_components=pca_result['components'],
            pca_mean=pca_result['mean'],
            batch_size=batch_size,
            pooling=pooling
        )

    elif method == "concat+pca+cluster":
        clusters, cluster_weights = compute_cluster_weights_for_pca(
            similarity_matrix, layer_quality, num_clusters=num_clusters
        )

        os.makedirs(pca_cache_dir, exist_ok=True)
        model_name_safe = model_name.replace("/", "_")
        pca_path = f"{pca_cache_dir}/{model_name_safe}_cluster{num_clusters}.pkl"

        if not os.path.exists(pca_path):
            logger.info(f"Training PCA for concat+pca+cluster ({num_clusters} clusters)...")
            pca_result = train_pca_for_clusters(
                model_name=model_name,
                clusters=clusters,
                cluster_weights=cluster_weights,
                output_path=pca_path,
                pooling=pooling
            )
        else:
            logger.info(f"Loading PCA from {pca_path}")
            with open(pca_path, 'rb') as f:
                pca_result = pickle.load(f)

        encoder = ClusterPCAEncoder(
            model_name=model_name,
            clusters=clusters,
            cluster_weights=cluster_weights,
            pca_components=pca_result['components'],
            pca_mean=pca_result['mean'],
            batch_size=batch_size,
            pooling=pooling
        )

    elif method == "concat+pca+all":
        os.makedirs(pca_cache_dir, exist_ok=True)
        model_name_safe = model_name.replace("/", "_")
        pca_path = f"{pca_cache_dir}/{model_name_safe}_all.pkl"

        if not os.path.exists(pca_path):
            logger.info(f"Training PCA for concat+pca+all (all layers)...")
            pca_result = train_pca_all_layers(
                model_name=model_name,
                output_path=pca_path,
                pooling=pooling
            )
        else:
            logger.info(f"Loading PCA from {pca_path}")
            with open(pca_path, 'rb') as f:
                pca_result = pickle.load(f)

        encoder = AllLayersPCAEncoder(
            model_name=model_name,
            pca_components=pca_result['components'],
            pca_mean=pca_result['mean'],
            batch_size=batch_size,
            pooling=pooling
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    unique_model_name = f"{model_name}_agg_{method}_lambda{lmbd}"
    encoder.model_name = unique_model_name

    return encoder


def save_intermediate_results(output_dir: str, config_name: str, results: Dict[str, Any]):
    """Save intermediate results after each task completion."""
    config_dir = Path(output_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    results_path = config_dir / "results_incremental.json"
    #print(results)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved intermediate results to {results_path}")




def extract_main_score(task_result: Dict[str, Any], task_name: str) -> float:
    """Extract the main evaluation score from a task result."""
    try:
        if task_result is None:
            return None

        scores = task_result.get('scores', {})
        if not scores:
            return None

        # Get the first split (usually 'test' or 'validation')
        split_name = list(scores.keys())[0]
        split_scores = scores[split_name]

        # Handle list format (MTEB returns list with one element)
        if isinstance(split_scores, list) and len(split_scores) > 0:
            split_scores = split_scores[0]

        # Return main_score if present
        if 'main_score' in split_scores:
            return split_scores['main_score']

        # Fallback to common metric names
        for metric in ['cosine_spearman', 'ndcg_at_10', 'accuracy', 
                       'v_measure', 'ap', 'cosine_pearson', 'map']:
            if metric in split_scores:
                return split_scores[metric]

        # If no known metric, return the first numeric value
        for key, value in split_scores.items():
            if isinstance(value, (int, float)) and key not in ['hf_subset', 'languages']:
                return value

        return None
    except Exception as e:
        logger.debug(f"Could not extract score for {task_name}: {e}")
        return None






def update_results_tables(output_dir: str, config_name: str, task_name: str, main_score):
    """
    Update both CSV and Markdown tables by appending/updating a single result.

    Args:
        output_dir: Directory to save tables
        config_name: Configuration name (e.g., "weighted_lambda0.5")
        task_name: Task name (e.g., "Banking77Classification")
        main_score: Score value (float, 'ERROR', or None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "results_table.csv"

    # Load existing data from CSV if it exists
    table_data = {}
    existing_configs = []

    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # First row is header: ['Task', 'config1', 'config2', ...]
                existing_configs = header[1:]  # Skip 'Task' column

                for row in reader:
                    if len(row) > 0:
                        task = row[0]
                        table_data[task] = {}
                        for i, config in enumerate(existing_configs):
                            if i + 1 < len(row):
                                value = row[i + 1]
                                if value == '':
                                    table_data[task][config] = None
                                elif value == 'ERROR':
                                    table_data[task][config] = 'ERROR'
                                else:
                                    try:
                                        table_data[task][config] = float(value)
                                    except ValueError:
                                        table_data[task][config] = value
        except Exception as e:
            logger.warning(f"Could not load existing CSV table: {e}. Starting fresh.")
            table_data = {}
            existing_configs = []

    # Add the new result
    if task_name not in table_data:
        table_data[task_name] = {}
    table_data[task_name][config_name] = main_score

    # Collect all unique config names
    all_configs = set(existing_configs)
    all_configs.add(config_name)
    config_names = sorted(all_configs)

    # Collect all task names
    task_names = sorted(table_data.keys())

    # Write CSV table
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Task'] + config_names)
            # Data rows
            for task in task_names:
                row = [task]
                for config in config_names:
                    value = table_data[task].get(config)
                    if value is None:
                        row.append('')
                    elif value == 'ERROR':
                        row.append('ERROR')
                    elif isinstance(value, (int, float)):
                        row.append(f'{value:.4f}')
                    else:
                        row.append(str(value))
                writer.writerow(row)
        logger.info(f"Updated CSV table: {csv_path} (task: {task_name}, config: {config_name})")
    except Exception as e:
        logger.error(f"Failed to write CSV table: {e}")

    # Write Markdown table
    md_path = output_path / "results_table.md"
    try:
        with open(md_path, 'w') as f:
            # Header
            f.write('| Task | ' + ' | '.join(config_names) + ' |\n')
            f.write('|' + '|'.join(['---'] * (len(config_names) + 1)) + '|\n')
            # Data rows
            for task in task_names:
                row = [task]
                for config in config_names:
                    value = table_data[task].get(config)
                    if value is None:
                        row.append('')
                    elif value == 'ERROR':
                        row.append('ERROR')
                    elif isinstance(value, (int, float)):
                        row.append(f'{value:.4f}')
                    else:
                        row.append(str(value))
                f.write('| ' + ' | '.join(row) + ' |\n')
        logger.info(f"Updated Markdown table: {md_path}")
    except Exception as e:
        logger.error(f"Failed to write Markdown table: {e}")


def run_evaluation(
    model_name: str,
    similarity_matrix_path: str,
    tasks: Optional[List[str]],
    task_types: Optional[List[str]],
    methods: List[str],
    lambda_values: List[float],
    output_dir: str,
    batch_size: int = 32,
    pca_cache_dir: str = "./pca_cache",
    pooling: str = "mean",
    num_clusters: int = 4,
    overwrite_results: bool = False,
    max_samples: Optional[int] = None
):
    """Run MTEB evaluation for all method/lambda combinations."""
    similarity_matrix = load_similarity_matrix(similarity_matrix_path)
    layer_quality = compute_layer_quality(similarity_matrix, method="diagonal")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading MTEB tasks...")

    if tasks and "all" in [t.lower() for t in tasks]:
        logger.info("Loading ALL available MTEB tasks")
        all_tasks = mteb.get_tasks(languages=["eng"])
    else:
        all_tasks = mteb.get_tasks(tasks=tasks, languages=["eng"])

    all_tasks = filter_text_only_tasks(all_tasks)

    if task_types:
        all_tasks = filter_by_task_types(all_tasks, task_types)

    
    if not all_tasks:
        logger.error("No valid text-only tasks remaining after filtering!")
        return

    results = {}

    for method in methods:
        for lmbd in lambda_values:
            config_name = f"{method}_lambda{lmbd}"
            logger.info("=" * 70)
            logger.info(f"Evaluating: {config_name}")
            logger.info("=" * 70)

            try:
                encoder = create_aggregated_encoder(
                    model_name=model_name,
                    similarity_matrix=similarity_matrix,
                    method=method,
                    layer_quality=layer_quality,
                    lmbd=lmbd,
                    batch_size=batch_size,
                    pca_cache_dir=pca_cache_dir,
                    pooling=pooling,
                    num_clusters=num_clusters
                )

                config_results = {
                    'method': method,
                    'lambda': lmbd,
                    'task_results': {}
                }

                for task in all_tasks:
                    task_name = task.metadata.name
                    logger.info(f"\n>>> Running task: {task_name}")
                    # NEW: Filter by sample count
                    if max_samples is not None:
                        res = filter_by_sample_count([task], max_samples)
                        if not res:
                            continue

                    try:
                        task_result = mteb.evaluate(
                            model=encoder,
                            tasks=[task],
                            encode_kwargs={"batch_size": batch_size},
                            show_progress_bar=True,
                            overwrite_strategy="always"  #'only-missing'
                        )
                        score = extract_main_score(task_result[0].to_dict(), task_name)
                        config_results['task_results'][task_name] = {
                            'status': 'completed',
                            'result': score
                        }

                        save_intermediate_results(output_dir, config_name, config_results)
                        update_results_tables(output_dir, config_name, task_name, score)
                        logger.info(f"✓ Completed {task_name}")

                    except Exception as task_error:
                        logger.error(f"✗ Failed {task_name}: {task_error}")
                        config_results['task_results'][task_name] = {
                            'status': 'failed',
                            'error': str(task_error)
                        }
                        save_intermediate_results(output_dir, config_name, config_results)
                        #update_results_tables(output_dir, config_name, task_name, score)

                results[config_name] = config_results
                logger.info(f"✓ Completed {config_name}")

            except Exception as e:
                logger.error(f"✗ Failed {config_name}: {e}", exc_info=True)
                results[config_name] = {'error': str(e)}

    summary_path = f"{output_dir}/evaluation_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info("=" * 70)
    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    logger.info(f"Summary: {summary_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with layer aggregation methods")

    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--similarity-matrix", type=str, required=True, help="Path to similarity matrix pickle")
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help='MTEB task names or "all"')
    parser.add_argument("--task-types", type=str, nargs="+", default=None,
                       choices=["Classification", "Clustering", "PairClassification", "Reranking", "Retrieval", "STS", "Summarization"],
                       help="Filter by task type")
    parser.add_argument("--methods", type=str, nargs="+", default=["weighted"],
                       choices=["weighted", "greedy", "greedyv2", "cluster", "concat+pca+qp", "concat+pca+cluster", "concat+pca+all"],
                       help="Aggregation methods")
    parser.add_argument("--lambda-values", type=float, nargs="+", default=[0.5], help="Lambda values")
    parser.add_argument("--pca-cache-dir", type=str, default="./pca_cache", help="PCA cache directory")
    parser.add_argument("--num-clusters", type=int, default=4, help="Number of clusters")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy")
    parser.add_argument("--overwrite-results", action="store_true", help="Overwrite cached results")

    # NEW: Filter by dataset size
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Maximum number of samples per task (for testing on small datasets)")

    args = parser.parse_args()

    run_evaluation(
        model_name=args.model_name,
        similarity_matrix_path=args.similarity_matrix,
        tasks=args.tasks,
        task_types=args.task_types,
        methods=args.methods,
        lambda_values=args.lambda_values,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pca_cache_dir=args.pca_cache_dir,
        pooling=args.pooling,
        num_clusters=args.num_clusters,
        overwrite_results=args.overwrite_results,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
