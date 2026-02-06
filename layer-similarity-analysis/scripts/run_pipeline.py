#!/usr/bin/env python3
"""
Main pipeline script for single experiments.

Usage:
    python scripts/run_pipeline.py \
        --model bert-base-uncased \
        --pooling mean \
        --dataset c4 \
        --output-dir ./results/bert_mean
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.extractor import EmbeddingExtractor
from src.embeddings.storage import save_embeddings
from src.data.loaders import load_dataset
from src.metrics.calculator import SimilarityCalculator, AVAILABLE_METRICS
from src.utils.logging import setup_logger
from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Layer Similarity Analysis Pipeline')

    # Model options
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model identifier')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'cls', 'last_token', 'max', 'pooler'],
                       help='Pooling strategy')

    # Data options
    parser.add_argument('--dataset', type=str, default='c4',
                       help='Dataset name (c4 or MTEB task)')
    parser.add_argument('--num-samples', type=int, default=20000,
                       help='Number of samples to process')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split')

    # Processing options
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding extraction')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision (FP16)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, default: auto)')

    # Similarity options
    parser.add_argument('--metrics', type=str, nargs='+', default=AVAILABLE_METRICS,
                       choices=AVAILABLE_METRICS,
                       help='Similarity metrics to calculate')
    parser.add_argument('--sample-size', type=int, default=3000,
                       help='Sample size for Monte Carlo trials')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of Monte Carlo trials')

    # Output options
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save embeddings to disk')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'pipeline_{timestamp}.log'
    logger = setup_logger('pipeline', log_file=log_file)

    logger.info("=" * 80)
    logger.info("LAYER SIMILARITY ANALYSIS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Pooling: {args.pooling}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    save_json(config, output_dir / 'config.json')

    # Step 1: Load data
    logger.info("\n[1/3] Loading data...")
    texts = load_dataset(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        split=args.split,
        seed=args.seed
    )
    logger.info(f"Loaded {len(texts)} texts")

    # Step 2: Extract embeddings
    logger.info("\n[2/3] Extracting embeddings...")
    extractor = EmbeddingExtractor(
        model_name=args.model,
        pooling_strategy=args.pooling,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.fp16
    )

    embeddings = extractor.extract_batch(
        texts=texts,
        batch_size=args.batch_size,
        show_progress=True
    )

    logger.info(f"Extracted embeddings for {len(embeddings)} layers")
    logger.info(f"Shape per layer: {embeddings[0].shape}")

    # Optionally save embeddings
    if args.save_embeddings:
        embeddings_dir = output_dir / 'embeddings'
        logger.info(f"Saving embeddings to {embeddings_dir}")
        save_embeddings(embeddings, embeddings_dir)

    # Step 3: Calculate similarities
    logger.info("\n[3/3] Calculating similarities...")
    calculator = SimilarityCalculator(
        sample_size=args.sample_size,
        num_trials=args.trials,
        seed=args.seed
    )

    results = calculator.calculate_all(
        embeddings=embeddings,
        metrics=args.metrics
    )

    # Save results
    similarities_dir = output_dir / 'similarities'
    ensure_dir(similarities_dir)

    for metric_name, metric_results in results.items():
        mean_matrix = metric_results['mean']
        std_matrix = metric_results['std']

        # Save as pickle
        with open(similarities_dir / f'{metric_name}.pkl', 'wb') as f:
            pickle.dump(mean_matrix, f)

        with open(similarities_dir / f'{metric_name}_std.pkl', 'wb') as f:
            pickle.dump(std_matrix, f)

        logger.info(f"Saved {metric_name}: shape {mean_matrix.shape}")

    # Save metadata
    metadata = {
        'model': args.model,
        'pooling': args.pooling,
        'dataset': args.dataset,
        'num_samples': len(texts),
        'num_layers': len(embeddings),
        'hidden_size': extractor.hidden_size,
        'metrics': args.metrics,
        'timestamp': timestamp
    }
    save_json(metadata, output_dir / 'metadata.json')

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
