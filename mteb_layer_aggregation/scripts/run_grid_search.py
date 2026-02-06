#!/usr/bin/env python
"""
Grid search script for computing similarity matrices across multiple configurations.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import logging
from src.similarity.grid_search import run_grid_search
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run grid search for similarity matrices")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./similarity_matrices",
        help="Output directory for similarity matrices"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples for similarity computation"
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Apply PCA before similarity computation"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("Starting grid search for similarity matrices")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")

    # Extract grid search parameters from config
    grid_config = config.get("grid_search", {})
    models = grid_config.get("models", [config["model"]["name"]])
    pooling_methods = grid_config.get("pooling_methods", [config["model"]["pooling"]])
    datasets = grid_config.get("datasets", [config["data"]["dataset"]])
    metrics = grid_config.get("similarity_metrics", config["similarity"]["metrics"])

    logger.info(f"Models: {models}")
    logger.info(f"Pooling methods: {pooling_methods}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Metrics: {metrics}")

    # Run grid search
    results = run_grid_search(
        models=models,
        pooling_methods=pooling_methods,
        datasets=datasets,
        similarity_metrics=metrics,
        n_samples=args.n_samples,
        use_pca=args.use_pca,
        output_dir=args.output_dir
    )

    logger.info("Grid search completed successfully")
    logger.info(f"Total experiments: {len(results)}")

    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\nExperiment {i}:")
        print(f"  Model: {result['model']}")
        print(f"  Pooling: {result['pooling']}")
        print(f"  Dataset: {result['dataset']}")
        print(f"  Similarity matrices saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
