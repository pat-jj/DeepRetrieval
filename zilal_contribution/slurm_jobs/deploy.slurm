#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=deploy
#SBATCH --output=logs/deploy_%j.out
#SBATCH --error=logs/deploy_%j.err
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

echo "=========================================="
echo " Starting Cluster infoat: $(date)"
echo " Running on host: $(hostname)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " GPUs requested: $SLURM_GPUS"
echo "=========================================="


# --- Activate conda environment ---
source /projects/illinois/eng/cs/jimeng/zelalae2/miniconda/etc/profile.d/conda.sh # adjust if different

conda activate zero


cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/original_deepRetrieval/DeepRetrieval

# --- Run your actual script ---
echo "Starting main script"
sh vllm_host.sh

echo "Finishing main script"


