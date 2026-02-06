# Layer Similarity Analysis - Complete Project

A modular Python framework for analyzing layer-wise representational similarity in transformer models. Refactored from Jupyter notebooks with systematic grid-search capabilities.

## 🎯 Features

- **Universal Model Support**: Works with any HuggingFace transformer (tested up to 3B parameters)
- **Multiple Pooling Strategies**: Mean, CLS, max, last token, trained pooler
- **Flexible Datasets**: C4 (universal) or any MTEB benchmark task
- **Multiple Similarity Metrics**: CKA, RSA, Jaccard Similarity, Distance Correlation
- **Grid Search**: Run all combinations in one command with automatic organization
- **Memory Efficient**: Batch processing with Monte Carlo sampling
- **Production Ready**: Logging, error handling, reproducible results

## 📁 Project Structure

```
layer-similarity-analysis/
├── src/
│   ├── models/
│   │   ├── pooling.py              # 5 pooling strategies
│   │   └── registry.py             # Model configuration
│   ├── embeddings/
│   │   ├── extractor.py            # Universal embedding extraction
│   │   └── storage.py              # Load/save utilities
│   ├── data/
│   │   └── loaders.py              # C4 + MTEB data loaders
│   ├── metrics/
│   │   └── calculator.py           # Similarity calculation (Monte Carlo)
│   ├── analysis/
│   │   └── visualization.py        # Heatmaps, dendrograms
│   └── utils/
│       ├── logging.py              # Logger setup
│       └── io.py                   # I/O utilities
├── scripts/
│   ├── run_pipeline.py             # ⭐ Unified pipeline (single experiment)
│   ├── run_grid_search.py          # ⭐ Grid search (multiple experiments)
│   └── run_batch_experiments.py    # YAML-based batch runner
├── experiments/
│   └── configs/                    # Example configurations
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Extract the downloaded zip file
unzip layer-similarity-analysis.zip
cd layer-similarity-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Analysis

```bash
# Quick test (< 5 minutes)
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --dataset c4 \
    --num-samples 1000 \
    --trials 3 \
    --output-dir ./test_results

# Full analysis
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --dataset c4 \
    --num-samples 20000 \
    --output-dir ./results/bert_mean
```

## 📊 Three Ways to Run Experiments

### 1. Single Experiment (run_pipeline.py)

Best for: Testing, single model analysis

```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --output-dir ./results/bert
```

### 2. Grid Search (run_grid_search.py) ⭐ RECOMMENDED

Best for: Systematic comparisons across models/pooling/datasets

```bash
# Compare 3 models × 2 pooling × 2 datasets = 12 experiments
python scripts/run_grid_search.py \
    --models bert-base-uncased roberta-base princeton-nlp/sup-simcse-roberta-base \
    --pooling mean cls \
    --datasets c4 Banking77Classification \
    --output-dir ./results/grid
```

**Automatic directory naming:**
- `results/bert_base_uncased_mean_c4/`
- `results/roberta_base_cls_banking77classification/`
- etc.

### 3. Batch Experiments (run_batch_experiments.py)

Best for: Custom configurations with fine-grained control

```bash
python scripts/run_batch_experiments.py \
    --config experiments/configs/model_comparison.yaml \
    --output-dir ./results/batch
```

## 💡 Common Use Cases

### Compare Different Models

```bash
python scripts/run_grid_search.py \
    --models bert-base-uncased roberta-base albert-base-v2 distilbert-base-uncased \
    --pooling mean \
    --datasets c4 \
    --output-dir ./results/model_comparison
```

### Test Pooling Strategies

```bash
python scripts/run_grid_search.py \
    --models bert-base-uncased \
    --pooling mean cls max last_token pooler \
    --datasets c4 \
    --output-dir ./results/pooling_comparison
```

### Analyze Dataset Effects

```bash
python scripts/run_grid_search.py \
    --models bert-base-uncased \
    --pooling mean \
    --datasets c4 Banking77Classification ImdbClassification EmotionClassification \
    --output-dir ./results/dataset_comparison
```

### Large Models (up to 3B parameters)

```bash
python scripts/run_grid_search.py \
    --models BAAI/bge-large-en-v1.5 intfloat/e5-large-v2 sentence-transformers/gtr-t5-xl \
    --pooling mean \
    --datasets c4 \
    --batch-size 16 \
    --fp16 \
    --output-dir ./results/large_models
```

## 📈 Output Structure

Each experiment creates:

```
output_dir/
├── similarities/
│   ├── CKA.pkl              # Mean similarity matrix [N × N]
│   ├── CKA_std.pkl          # Standard deviation matrix
│   ├── RSA.pkl
│   ├── RSA_std.pkl
│   ├── JaccardSimilarity.pkl
│   └── DistanceCorrelation.pkl
├── embeddings/              # Optional (if --save-embeddings)
│   ├── layer_0/
│   │   ├── batch_0.pt
│   │   └── ...
│   └── ...
├── metadata.json            # Full experiment configuration
└── pipeline_*.log           # Detailed execution log
```

Grid searches also create:
```
grid_results/
├── bert_base_uncased_mean_c4/
├── roberta_base_mean_c4/
├── ...
└── grid_search_summary.json  # Summary of all experiments
```

## 🔧 Configuration Options

### Model Options
- `--model`: HuggingFace model identifier (required)
- `--pooling`: Pooling strategy (default: mean)
  - Choices: `mean`, `cls`, `last_token`, `max`, `pooler`

### Data Options
- `--dataset`: Dataset name (default: c4)
  - Options: `c4` or any MTEB task name
- `--num-samples`: Number of samples (default: 20000)
- `--split`: Dataset split for MTEB (default: train)

### Processing Options
- `--batch-size`: Batch size for extraction (default: 64)
- `--max-length`: Max sequence length (default: 256)
- `--fp16`: Use mixed precision (faster, less memory)

### Similarity Options
- `--metrics`: Metrics to calculate (default: all 4)
  - Options: `CKA`, `RSA`, `JaccardSimilarity`, `DistanceCorrelation`
- `--sample-size`: Samples per trial (default: 3000)
- `--trials`: Monte Carlo trials (default: 5)

### Output Options
- `--output-dir`: Output directory (required)
- `--save-embeddings`: Save embeddings to disk
- `--seed`: Random seed (default: 42)

## 📊 Loading and Analyzing Results

```python
import pickle
import json
import numpy as np

# Load similarity matrix
with open('results/bert_mean/similarities/CKA.pkl', 'rb') as f:
    cka_matrix = pickle.load(f)

# Load metadata
with open('results/bert_mean/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model: {metadata['model']}")
print(f"Layers: {metadata['num_layers']}")
print(f"CKA shape: {cka_matrix.shape}")

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(cka_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('CKA Layer Similarity')
plt.xlabel('Layer')
plt.ylabel('Layer')
plt.tight_layout()
plt.savefig('cka_heatmap.png', dpi=300)
plt.show()
```

## ⚡ Performance Tips

1. **Use --fp16** for large models (2x faster, 2x less memory)
2. **Adjust batch-size** based on GPU memory:
   - BERT-base: 64-128
   - Large models: 16-32
   - 3B models: 4-8
3. **Use --dry-run** first to check grid search configurations
4. **Enable --continue-on-error** for long-running grid searches
5. **Start with quick test**: 1000 samples, 3 trials, CKA only

## 🔍 Differences from Original Notebooks

### Improvements
✅ **Unified pipeline**: One command for extraction + calculation  
✅ **Model agnostic**: Works with any HuggingFace transformer  
✅ **Multiple pooling**: 5 strategies vs 2 in notebooks  
✅ **Dataset flexibility**: Easy switching between C4 and MTEB  
✅ **Grid search**: Systematic comparisons in one command  
✅ **No duplication**: Single codebase vs 2 notebooks  
✅ **Fixed bugs**: Corrected file sorting issue from SimCSE notebook  
✅ **Clean code**: Removed unused MTEB exploration code  
✅ **Production ready**: Logging, error handling, testing structure  

### What's Preserved
✓ Same similarity metrics (CKA, RSA, Jaccard, Distance Correlation)  
✓ Same Monte Carlo sampling approach  
✓ Same pooling implementations  
✓ Compatible output format (can load old .pkl files)  

## 📝 Example Workflows

### Workflow 1: Quick Test

```bash
# Test the pipeline (< 5 minutes)
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --num-samples 1000 \
    --trials 3 \
    --metrics CKA \
    --output-dir ./test
```

### Workflow 2: Reproduce Notebooks

```bash
python scripts/run_grid_search.py \
    --models bert-base-uncased princeton-nlp/sup-simcse-roberta-base \
    --pooling mean cls pooler \
    --datasets c4 \
    --num-samples 20000 \
    --output-dir ./notebook_reproduction
```

### Workflow 3: Large-Scale Analysis

```bash
# 5 models × 3 pooling × 2 datasets = 30 experiments
python scripts/run_grid_search.py \
    --models \
        bert-base-uncased \
        roberta-base \
        albert-base-v2 \
        princeton-nlp/sup-simcse-roberta-base \
        BAAI/bge-large-en-v1.5 \
    --pooling mean cls max \
    --datasets c4 Banking77Classification \
    --fp16 \
    --continue-on-error \
    --output-dir ./large_scale_analysis
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Enable `--fp16`
- Reduce `--num-samples` or `--sample-size`

### Slow Execution
- Enable `--fp16`
- Increase `--batch-size` (if memory allows)
- Reduce `--trials` (minimum: 3)

### Model Not Found
- Check HuggingFace model name
- Ensure model supports `output_hidden_states=True`

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version >= 3.8

## 📚 Citation

If you use this framework, please cite:
```bibtex
@software{layer_similarity_analysis,
  title={Layer Similarity Analysis Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/layer-similarity-analysis}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Created**: February 2026  
**Based on**: Original RoBERTa and SimCSE similarity analysis notebooks  
**Version**: 1.0.0
