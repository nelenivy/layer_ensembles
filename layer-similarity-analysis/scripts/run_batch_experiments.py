#!/usr/bin/env python3
"""
Batch experiment runner using YAML configuration.

Usage:
    python scripts/run_batch_experiments.py \
        --config experiments/configs/my_experiment.yaml \
        --output-dir ./results/batch
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch Experiment Runner'
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Base output directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment_from_config(
    exp_config: dict,
    output_dir: Path,
    dry_run: bool = False
) -> bool:
    """Run experiment from configuration dict."""
    cmd = ['python', 'scripts/run_pipeline.py']

    # Add all parameters from config
    for key, value in exp_config.items():
        if key == 'name':
            continue

        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif isinstance(value, list):
            cmd.extend([arg_name] + [str(v) for v in value])
        else:
            cmd.extend([arg_name, str(value)])

    # Add output directory
    cmd.extend(['--output-dir', str(output_dir)])

    print(f"\nCommand: {' '.join(cmd)}")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main():
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Create base output directory
    base_output_dir = Path(args.output_dir)
    ensure_dir(base_output_dir)

    print("=" * 80)
    print("BATCH EXPERIMENTS")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {base_output_dir}")
    print(f"Experiments: {len(config.get('experiments', []))}")
    print("=" * 80)

    # Run experiments
    results = []
    for i, exp_config in enumerate(config.get('experiments', []), 1):
        exp_name = exp_config.get('name', f'experiment_{i}')
        exp_output_dir = base_output_dir / exp_name

        print(f"\n[{i}/{len(config['experiments'])}] Running: {exp_name}")

        success = run_experiment_from_config(
            exp_config,
            exp_output_dir,
            dry_run=args.dry_run
        )

        results.append({
            'name': exp_name,
            'success': success
        })

    # Save summary
    if not args.dry_run:
        summary = {
            'config_file': str(config_path),
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        save_json(summary, base_output_dir / 'batch_summary.json')

    print("\n" + "=" * 80)
    print("BATCH COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
