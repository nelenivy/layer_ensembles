# MTEB Layer Aggregation Experiments

Comprehensive framework for evaluating layer aggregation strategies in NLP models on the MTEB benchmark.

## Features

✅ **Unified Codebase** - Eliminates all duplications from original notebooks
✅ **Modern Model Support** - Works with any HuggingFace transformer up to 3B parameters  
✅ **All MTEB Tasks** - Automatic evaluation on all 58+ MTEB tasks
✅ **Multiple Aggregation Methods** - 6+ aggregation strategies implemented
✅ **Pipeline Integration** - Seamlessly integrates with similarity calculation pipeline
✅ **Memory Efficient** - Incremental PCA, batch processing, GPU cleanup

## Project Structure

```
mteb_layer_aggregation/
├── src/
│   ├── models/          # Encoder implementations
│   ├── aggregation/     # Aggregation methods & optimization
│   ├── data/            # Data loaders & MTEB tasks
│   ├── evaluation/      # Evaluation runners
│   ├── metrics/         # Similarity metrics
│   └── utils/           # Utilities
├── scripts/             # Execution scripts
├── configs/             # YAML configurations
└── requirements.txt
```

## Installation

```bash
# Clone or extract the project
cd mteb_layer_aggregation

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### 1. Run Complete Pipeline (Recommended)

python scripts/run_mteb_eval.py     --model-name bert-base-uncased     --similarity-matrix /home/jovyan/shestov/layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl     --tasks all   --task-types Classification     --lambda-values 0.1 0.5 1.0  --methods weighted greedy cluster greedyv2      --output-dir ./aggregation_results16 --max-samples 10000 --batch-size 512

# NEURSLOP
Runs similarity calculation → aggregation → evaluation on all MTEB tasks:

```bash
python scripts/run_full_pipeline.py \
    --models bert-base-uncased roberta-base sentence-transformers/all-MiniLM-L6-v2 \
    --output-dir ./results \
    --all-mteb-tasks
```

### 2. Run Aggregation Experiments Only

Use pre-computed similarity matrices:

```bash
python scripts/run_aggregation_experiments.py \
    --model-name bert-base-uncased \
    --similarity-matrix ./similarity_results/bert_base_uncased_mean_c4/CKA.pkl \
    --tasks Banking77Classification ImdbClassification NFCorpus \
    --methods weighted greedy cluster_pca selected_pca \
    --output-dir ./aggregation_results
```

### 3. Run Grid Search for Similarities

Calculate similarity matrices for multiple models:

```bash
python scripts/run_grid_search.py \
    --models bert-base-uncased roberta-base \
    --pooling mean cls \
    --datasets c4 \
    --num-samples 10000 \
    --output-dir ./similarity_results
```

## Aggregation Methods

### 1. **Weighted Mean** (`weighted`)
Weighted average of selected layers using quadratic programming optimization.

### 2. **Greedy Selection** (`greedy`)
Greedy layer selection balancing quality and diversity.

### 3. **Cluster + PCA** (`cluster_pca`)
Cluster layers by similarity, concatenate cluster means, apply PCA.

### 4. **Selected Layers + PCA** (`selected_pca`)
Select best layers, concatenate, apply PCA projection.

### 5. **All Layers + PCA** (`all_pca`)
Concatenate all layers with PCA dimensionality reduction.

### 6. **Best Single Layer** (`best_single`)
Use single best performing layer.

## Configuration

### Model Configuration (`configs/models_config.yaml`)

```yaml
models:
  - name: bert-base-uncased
    max_length: 512
    batch_size: 32

  - name: roberta-base
    max_length: 512
    batch_size: 32

  - name: sentence-transformers/all-MiniLM-L6-v2
    max_length: 256
    batch_size: 64
```

### Aggregation Configuration (`configs/aggregation_config.yaml`)

```yaml
aggregation:
  methods:
    - weighted
    - greedy
    - cluster_pca
    - selected_pca

  optimization:
    lambda_penalty: 0.5
    drop_delta: 0.15
    max_layers: null

  pca:
    output_dim: 768
    n_sentences: 30000
    batch_size: 256
```

### Experiment Configuration (`configs/experiment_config.yaml`)

```yaml
experiment:
  similarity_metrics:
    - CKA
    - RSA
    - correlation

  mteb_categories:
    - Classification
    - Clustering
    - Retrieval
    - STS
    - PairClassification
    - Reranking

  evaluation:
    batch_size: 32
    languages: [eng]
```

## Advanced Usage

### Evaluate Specific Task Categories

```bash
python scripts/run_aggregation_experiments.py \
    --model-name bert-base-uncased \
    --similarity-matrix ./results/CKA.pkl \
    --task-categories Classification Retrieval \
    --methods greedy cluster_pca
```

### Use Custom Similarity Matrix

```bash
python scripts/run_aggregation_experiments.py \
    --model-name your-model-name \
    --similarity-matrix /path/to/custom_similarity.pkl \
    --similarity-type CKA \
    --all-mteb-tasks
```

### Export Results

Results are automatically saved in multiple formats:
- `results.json` - Complete evaluation metrics
- `summary.csv` - Tabular summary across tasks
- `checkpoints/*.pt` - PCA components and weights
- `logs/experiment.log` - Detailed execution log

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if running out of memory
2. **Large Models**: Use `--fp16` flag for memory efficiency
3. **PCA Training**: Increase `n_sentences` for better PCA quality
4. **Parallel Execution**: Use multiple GPUs with `CUDA_VISIBLE_DEVICES`

## Output Structure

```
results/
├── bert_base_uncased/
│   ├── weighted/
│   │   ├── Banking77Classification/
│   │   │   ├── results.json
│   │   │   └── metrics.csv
│   │   └── checkpoint.pt
│   ├── greedy/
│   ├── cluster_pca/
│   └── summary.csv
└── experiment_summary.json
```

## Integration with Existing Pipeline

This project reads similarity matrices from your existing grid search pipeline:

```python
# Your grid search creates:
./similarity_results/bert_base_uncased_mean_c4/CKA.pkl

# This project reads and uses them:
python scripts/run_aggregation_experiments.py \
    --similarity-matrix ./similarity_results/bert_base_uncased_mean_c4/CKA.pkl
```

## Supported Models

- **BERT family**: bert-base-uncased, bert-large-uncased, etc.
- **RoBERTa family**: roberta-base, roberta-large, etc.
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
- **Modern embedders**: BAAI/bge-*, nomic-ai/nomic-embed-*, etc.
- **Any HuggingFace transformer** up to 3B parameters

## MTEB Task Categories Supported

- Classification (12 tasks)
- Clustering (11 tasks)
- Pair Classification (3 tasks)
- Reranking (4 tasks)
- Retrieval (15 tasks)
- STS (10 tasks)
- Summarization (1 task)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mteb_layer_aggregation,
  title={MTEB Layer Aggregation Framework},
  year={2026},
  author={Research Team}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Troubleshooting

**Issue**: Out of GPU memory  
**Solution**: Reduce `batch_size` or use `--fp16`

**Issue**: Similarity matrix not found  
**Solution**: Check file path and naming convention matches grid search output

**Issue**: Task evaluation fails  
**Solution**: Ensure MTEB and datasets are up to date: `pip install --upgrade mteb datasets`

## Contact

For questions or issues, please open an issue on GitHub.
