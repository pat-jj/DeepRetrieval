DATA_PATH=data/matching/qwen-instruct/test.parquet
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
MODEL_NAME=qwen2.5-3B-instruct

CUDA_VISIBLE_DEVICES=1 python src/eval/eval_inst.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
