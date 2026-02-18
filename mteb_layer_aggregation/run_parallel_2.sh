# Kill all child processes on script termination
trap 'pkill -P $$; exit' SIGINT SIGTERM EXIT

CUDA_VISIBLE_DEVICES=2 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --tasks core \
    --lambda-values 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache2" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > bert_mean8_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --tasks core \
    --lambda-values 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --per-dataset-similarity-metric CKA \
    --per-dataset-hessian \
    --hessian-cache-dir "./hessian_cache2" \
    --similarity-cache-dir "./similarity_cache2" \
    --use-embedding-cache --use-quality-cache > simcse_mean8_output.log 2>&1 &


# Wait for both to complete
wait
