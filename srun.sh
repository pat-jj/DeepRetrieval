source activate zero

srun --job-name="lang_task" \
     --account=bdgw-delta-gpu \
     --partition=gpuA40x4,gpuA100x4,gpuA100x8 \
     --mem=64G \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --constraint="scratch" \
     --gpus-per-node=1 \
     --gpu-bind=closest \
     --time=8:00:00 \
     python eval/baseline.py


# srun --job-name="lang_task" \
#      --account=bdgw-delta-gpu \
#      --partition=gpuA40x4,gpuA100x4,gpuA100x8 \
#      --mem=64G \
#      --nodes=1 \
#      --ntasks-per-node=1 \
#      --cpus-per-task=16 \
#      --constraint="scratch" \
#      --gpus-per-node=1 \
#      --gpu-bind=closest \
#      --time=8:00:00 \
#      python test.py