export CUDA_VISIBLE_DEVICES=3

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/nq.py \
    --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_1150 \
    --model_name nq-3b-step-1150