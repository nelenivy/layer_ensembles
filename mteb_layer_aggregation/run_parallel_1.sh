# Kill all child processes on script termination
trap 'pkill -P $$; exit' SIGINT SIGTERM EXIT

CUDA_VISIBLE_DEVICES=0 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods weighted_greedy \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > bert_mean5_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods weighted_greedy \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > simcse_mean5_output.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods greedy_rank \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > bert_mean_7_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods greedy_rank \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > simcse_mean_7_output.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2  python scripts/run_mteb_eval.py \
#     --model-name princeton-nlp/sup-simcse-roberta-base \
#     --tasks core \
#     --lambda-values 0.01 5.0 \
#     --methods greedy_rank \
#     --output-dir ./aggregation_results_simcse_mean_dis \
#     --max-samples 10000000 --batch-size 512 \
#     --per-dataset-similarity-metric CKA \
#     --similarity-cache-dir "./similarity_cache2" \
#     --normalize-layer-quality \
#     --use-log-similarity \
#     --use-embedding-cache --use-quality-cache > simcse_mean_8_output.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python scripts/run_mteb_eval.py \
#     --model-name bert-base-uncased \
#     --tasks core \
#     --lambda-values 0.01 5.0 \
#     --methods greedy_rank  \
#     --output-dir ./aggregation_results_bert_mean_dis \
#     --max-samples 10000000 --batch-size 512 \
#     --per-dataset-similarity-metric CKA \
#     --similarity-cache-dir "./similarity_cache2" \
#     --normalize-layer-quality \
#     --use-log-similarity \
#     --use-embedding-cache --use-quality-cache > bert_mean_8_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > bert_mean_9_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --tasks core \
    --lambda-values 0.01 0.1 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache1" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > simcse_mean_9_output.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2  python scripts/run_mteb_eval.py \
#     --model-name princeton-nlp/sup-simcse-roberta-base \
#     --tasks core \
#     --lambda-values 0.1 0.5 1.0 \
#     --methods weighted \
#     --output-dir ./aggregation_results_simcse_mean_dis \
#     --max-samples 10000000 --batch-size 512 \
#     --per-dataset-similarity-metric CKA \
#     --similarity-cache-dir "./similarity_cache2" \
#     --normalize-layer-quality  \
#     --use-log-similarity \
#     --use-embedding-cache --use-quality-cache > simcse_mean_10_output.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python scripts/run_mteb_eval.py \
#     --model-name bert-base-uncased \
#     --tasks core \
#     --lambda-values 0.1 0.5 1.0 \
#     --methods weighted \
#     --output-dir ./aggregation_results_bert_mean_dis \
#     --max-samples 10000000 --batch-size 512 \
#     --per-dataset-similarity-metric CKA \
#     --similarity-cache-dir "./similarity_cache2" \
#     --normalize-layer-quality  \
#     --use-log-similarity \
#     --use-embedding-cache --use-quality-cache > bert_mean_10_output.log 2>&1 &

# Wait for both to complete
wait
