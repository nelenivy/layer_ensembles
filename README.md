## Pipeline

### Install layer similarity calculation

```bash
cd ../layer-similarity-analysis

pip install -e .
```

### Run grid over different base LMs and pooling strategies over a Common Crawl C4 dataset on a small sample

```bash
python scripts/run_grid_search.py     --models bert-base-uncased roberta-base princeton-nlp/sup-simcse-roberta-base     --pooling mean cls last_token     --datasets c4     --output-dir ./results/grid
```

### Install ensembles construction and evaluation

```bash
cd ../mteb_layer_aggregation

pip install -e .
```

### Run evaluation of different layer ensembles methods on MTEB on classificain tasks on small datasets < 10k with precomputed similarity matrix with BERT as base

```bash
python scripts/run_mteb_eval.py     --model-name bert-base-uncased     --similarity-matrix /home/jovyan/shestov/layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl     --tasks all   --task-types Classification     --lambda-values 0.1 0.5 1.0  --methods weighted greedy cluster greedyv2      --output-dir ./aggregation_results16 --max-samples 10000 --batch-size 512
```