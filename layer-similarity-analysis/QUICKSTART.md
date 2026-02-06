# Quick Start Guide

Get up and running in 5 minutes!

## Installation

```bash
# Clone and navigate
cd layer-similarity-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Your First Experiment

### 1. Quick Test (1-2 minutes)

```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --dataset c4 \
    --num-samples 1000 \
    --trials 3 \
    --output-dir ./test_run
```

### 2. Full Single Experiment (10-15 minutes)

```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --dataset c4 \
    --num-samples 20000 \
    --output-dir ./results/bert_mean
```

### 3. Grid Search (Recommended for research)

```bash
# Compare 2 models with 2 pooling strategies
python scripts/run_grid_search.py \
    --models bert-base-uncased roberta-base \
    --pooling mean cls \
    --datasets c4 \
    --output-dir ./results/grid

# This creates 4 experiments:
# - bert_base_uncased_mean_c4
# - bert_base_uncased_cls_c4
# - roberta_base_mean_c4
# - roberta_base_cls_c4
```

## Analyze Results

```python
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load similarity matrix
with open('results/bert_mean/similarities/CKA.pkl', 'rb') as f:
    cka_matrix = pickle.load(f)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cka_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('BERT Layer Similarity (CKA)')
plt.xlabel('Layer')
plt.ylabel('Layer')
plt.tight_layout()
plt.savefig('bert_cka.png', dpi=300)
plt.show()
```

## Common Options

### Speed up with FP16
```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --fp16  # 2x faster!
```

### Calculate specific metrics only
```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --metrics CKA RSA  # Only CKA and RSA
```

### Adjust batch size for your GPU
```bash
python scripts/run_pipeline.py \
    --model bert-base-uncased \
    --pooling mean \
    --batch-size 32  # Reduce if OOM
```

## Next Steps

- See `README.md` for comprehensive documentation
- Check `experiments/configs/` for example YAML configs
- Try different models from HuggingFace
- Experiment with different pooling strategies
- Compare results across datasets

## Troubleshooting

**Out of memory?**
- Use `--fp16`
- Reduce `--batch-size`
- Reduce `--num-samples`

**Slow?**
- Enable `--fp16`
- Increase `--batch-size` (if memory allows)
- Reduce `--trials`

**Import errors?**
- Check: `pip install -r requirements.txt`
- Ensure Python >= 3.8
