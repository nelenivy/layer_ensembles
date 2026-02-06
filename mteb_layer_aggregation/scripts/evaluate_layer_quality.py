#!/usr/bin/env python
"""
Script for evaluating layer quality before aggregation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import logging
import pickle
from src.evaluation.evaluators import evaluate_all_layers, extract_layer_quality
from src.data.loaders import load_classification_dataset, load_retrieval_dataset
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate layer quality")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["classification", "retrieval", "sts"],
        help="Type of evaluation task"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., mteb/banking77)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./layer_quality.pkl",
        help="Output path for layer quality scores"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("Evaluating layer quality")
    logger.info(f"Task type: {args.task_type}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {config['model']['name']}")

    # Load data based on task type
    if args.task_type == "classification":
        ds = load_classification_dataset(args.dataset)
        data = {
            "train_texts": ds["train"]["text"],
            "train_labels": ds["train"]["label"],
            "val_texts": ds["dev"]["text"],
            "val_labels": ds["dev"]["label"]
        }
        metric_name = "accuracy"

    elif args.task_type == "retrieval":
        queries, corpus, relevance = load_retrieval_dataset(args.dataset)
        data = {
            "queries": queries,
            "corpus": corpus,
            "relevance": relevance
        }
        metric_name = "recall@10"

    elif args.task_type == "sts":
        # STS datasets need custom loading based on format
        raise NotImplementedError("STS evaluation not fully implemented in this script")

    # Evaluate all layers
    logger.info("Evaluating all layers...")
    results = evaluate_all_layers(
        model_name=config["model"]["name"],
        task_type=args.task_type,
        data=data,
        pooling=config["model"]["pooling"],
        batch_size=args.batch_size
    )

    # Extract quality scores
    quality_scores = extract_layer_quality(results, metric=metric_name)

    # Print results
    print("\n" + "="*80)
    print("LAYER QUALITY EVALUATION")
    print("="*80)
    print(f"\nTask: {args.dataset}")
    print(f"Metric: {metric_name}")
    print("\nScores by layer:")
    for layer_idx, score in enumerate(quality_scores):
        print(f"  Layer {layer_idx}: {score:.4f}")

    print(f"\nBest layer: {quality_scores.argmax()} (score: {quality_scores.max():.4f})")
    print(f"Worst layer: {quality_scores.argmin()} (score: {quality_scores.min():.4f})")
    print(f"Average: {quality_scores.mean():.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
