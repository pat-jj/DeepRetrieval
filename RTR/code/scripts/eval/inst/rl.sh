DATA_PATH=data/matching/qwen-instruct/test.parquet
MODEL_PATH=checkpoints/Panacea-Zero/matching-qwen2.5-3b-inst-ppo-2gpus/actor/global_step_400
MODEL_NAME=ours-inst-ppo

CUDA_VISIBLE_DEVICES=0 python src/eval/eval_inst.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
