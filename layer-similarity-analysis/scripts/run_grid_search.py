#!/usr/bin/env python3
"""
Grid search script for systematic experiments.

Runs all combinations of models, pooling strategies, and datasets.
Automatically names output directories based on configuration.

Usage:
    python scripts/run_grid_search.py \
        --models bert-base-uncased roberta-base \
        --pooling mean cls \
        --datasets c4 Banking77Classification \
        --output-dir ./results/grid
"""

import argparse
import subprocess
import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import ensure_dir, save_json


def sanitize_name(name: str) -> str:
    """Sanitize name for directory/file naming."""
    return name.replace('/', '_').replace('-', '_').lower()


def generate_experiment_name(model: str, pooling: str, dataset: str) -> str:
    """Generate experiment directory name."""
    model_name = sanitize_name(model.split('/')[-1])
    dataset_name = sanitize_name(dataset)
    return f"{model_name}_{pooling}_{dataset_name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid Search for Layer Similarity Analysis'
    )

    # Grid parameters
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='List of HuggingFace model identifiers')
    parser.add_argument('--pooling', type=str, nargs='+', 
                       default=['mean'],
                       choices=['mean', 'cls', 'last_token', 'max', 'pooler'],
                       help='List of pooling strategies')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['c4'],
                       help='List of dataset names')

    # Shared parameters for all experiments
    parser.add_argument('--num-samples', type=int, default=20000,
                       help='Number of samples per experiment')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Max sequence length')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Metrics to calculate (default: all)')
    parser.add_argument('--sample-size', type=int, default=3000,
                       help='Monte Carlo sample size')
    parser.add_argument('--trials', type=int, default=5,
                       help='Monte Carlo trials')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output options
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Base output directory')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save embeddings for all experiments')

    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue if experiment fails')

    return parser.parse_args()


def run_experiment(
    model: str,
    pooling: str,
    dataset: str,
    output_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False
) -> bool:
    """
    Run single experiment.

    Returns:
        True if successful, False if failed
    """
    # Build command
    cmd = [
        'python', 'scripts/run_pipeline.py',
        '--model', model,
        '--pooling', pooling,
        '--dataset', dataset,
        '--num-samples', str(args.num_samples),
        '--batch-size', str(args.batch_size),
        '--max-length', str(args.max_length),
        '--sample-size', str(args.sample_size),
        '--trials', str(args.trials),
        '--seed', str(args.seed),
        '--output-dir', str(output_dir)
    ]

    if args.fp16:
        cmd.append('--fp16')

    if args.save_embeddings:
        cmd.append('--save-embeddings')

    if args.metrics:
        cmd.extend(['--metrics'] + args.metrics)

    # Print command
    print(f"\n{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    if dry_run:
        return True

    # Execute
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with code {e.returncode}")
        return False


def main():
    args = parse_args()

    # Create base output directory
    base_output_dir = Path(args.output_dir)
    ensure_dir(base_output_dir)

    # Generate all combinations
    combinations = list(product(args.models, args.pooling, args.datasets))
    total_experiments = len(combinations)

    print("=" * 80)
    print("GRID SEARCH CONFIGURATION")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Pooling: {args.pooling}")
    print(f"Datasets: {args.datasets}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 80)

    # Show all experiment names
    print("\nExperiment directories:")
    for model, pooling, dataset in combinations:
        exp_name = generate_experiment_name(model, pooling, dataset)
        print(f"  - {exp_name}")

    # Confirmation
    if not args.dry_run:
        print(f"\nThis will run {total_experiments} experiments.")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run experiments
    results = []
    successful = 0
    failed = 0

    for i, (model, pooling, dataset) in enumerate(combinations, 1):
        exp_name = generate_experiment_name(model, pooling, dataset)
        exp_output_dir = base_output_dir / exp_name

        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {i}/{total_experiments}: {exp_name}")
        print(f"{'#'*80}")

        # Run experiment
        success = run_experiment(
            model=model,
            pooling=pooling,
            dataset=dataset,
            output_dir=exp_output_dir,
            args=args,
            dry_run=args.dry_run
        )

        # Record result
        result_entry = {
            'experiment_name': exp_name,
            'model': model,
            'pooling': pooling,
            'dataset': dataset,
            'output_dir': str(exp_output_dir),
            'success': success
        }
        results.append(result_entry)

        if success:
            successful += 1
            print(f"✓ Experiment {i}/{total_experiments} completed successfully")
        else:
            failed += 1
            print(f"✗ Experiment {i}/{total_experiments} failed")

            if not args.continue_on_error:
                print("\nStopping due to error (use --continue-on-error to continue)")
                break

    # Save summary
    if not args.dry_run:
        summary = {
            'total_experiments': total_experiments,
            'successful': successful,
            'failed': failed,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'models': args.models,
                'pooling': args.pooling,
                'datasets': args.datasets,
                'num_samples': args.num_samples
            },
            'results': results
        }

        summary_file = base_output_dir / 'grid_search_summary.json'
        save_json(summary, summary_file)

        print("\n" + "=" * 80)
        print("GRID SEARCH COMPLETED")
        print("=" * 80)
        print(f"Total: {total_experiments}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Summary saved to: {summary_file}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETED")
        print("=" * 80)
        print(f"Would run {total_experiments} experiments")
        print("Remove --dry-run to execute")
        print("=" * 80)


if __name__ == '__main__':
    main()
