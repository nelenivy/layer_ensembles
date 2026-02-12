# Kill all child processes on script termination
trap 'pkill -P $$; exit' SIGINT SIGTERM EXIT

CUDA_VISIBLE_DEVICES=0 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --similarity-matrix ../layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted_greedy greedy_rank  weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-softmax \
    --use-embedding-cache --use-quality-cache > bert_mean_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --similarity-matrix ../layer-similarity-analysis/results/grid/sup_simcse_roberta_base_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted_greedy greedy_rank  weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-softmax \
    --use-embedding-cache --use-quality-cache > simcse_mean_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=2  python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --similarity-matrix ../layer-similarity-analysis/results/grid/sup_simcse_roberta_base_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted_greedy greedy_rank weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-log-quality \
    --use-embedding-cache --use-quality-cache > simcse_mean_2_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --similarity-matrix ../layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted_greedy greedy_rank  weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-log-quality \
    --use-embedding-cache --use-quality-cache > bert_mean_2_output.log 2>&1 &


UDA_VISIBLE_DEVICES=0 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --similarity-matrix ../layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-softmax \
    --use-embedding-cache --use-quality-cache > bert_mean_3_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --similarity-matrix ../layer-similarity-analysis/results/grid/sup_simcse_roberta_base_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-softmax \
    --use-embedding-cache --use-quality-cache > simcse_mean_3_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=2  python scripts/run_mteb_eval.py \
    --model-name princeton-nlp/sup-simcse-roberta-base \
    --similarity-matrix ../layer-similarity-analysis/results/grid/sup_simcse_roberta_base_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_simcse_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-log-quality \
    --use-embedding-cache --use-quality-cache > simcse_mean_4_output.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/run_mteb_eval.py \
    --model-name bert-base-uncased \
    --similarity-matrix ../layer-similarity-analysis/results/grid/bert_base_uncased_mean_c4/similarities/CKA.pkl \
    --tasks core \
    --lambda-values 0.1 0.5 1.0 \
    --methods weighted \
    --output-dir ./aggregation_results_bert_mean \
    --max-samples 10000000 --batch-size 512 \
    --use-log-quality \
    --use-embedding-cache --use-quality-cache > bert_mean_4_output.log 2>&1 &

# Wait for both to complete
wait
