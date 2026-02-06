# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-04

### Added
- Initial release of Layer Similarity Analysis framework
- Universal embedding extraction for any HuggingFace transformer
- 5 pooling strategies: mean, CLS, max, last_token, pooler
- 4 similarity metrics: CKA, RSA, Jaccard Similarity, Distance Correlation
- Monte Carlo sampling for efficient similarity calculation
- Grid search functionality for systematic experiments
- Automatic directory naming based on configuration
- C4 dataset support
- MTEB dataset integration
- FP16 mixed precision support
- Comprehensive logging system
- Visualization utilities (heatmaps, dendrograms)
- YAML-based batch experiment runner
- Example configurations for common use cases
- Complete documentation and quick start guide

### Features
- Single-command pipeline for extraction + calculation
- Multi-model, multi-pooling, multi-dataset grid search
- Reproducible experiments with seed control
- Memory-efficient batch processing
- Error handling and retry logic
- Progress bars and status updates
- JSON metadata export
- Pickle-based result storage

### Documentation
- Comprehensive README with examples
- Quick start guide for beginners
- Troubleshooting section
- API documentation in docstrings
- Example YAML configurations
