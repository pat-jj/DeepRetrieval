#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=lora_pubmed_32
#SBATCH --output=logs/lora_pubmed_32_%j.out
#SBATCH --error=logs/lora_pubmed_32_%j.err
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

echo "=========================================="
echo " Starting Cluster infoat: $(date)"
echo " Running on host: $(hostname)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " GPUs requested: $SLURM_GPUS"
echo "=========================================="

# --- Load CUDA module ---
module load cuda/12.6

# --- Activate conda environment ---
source ~/miniconda/etc/profile.d/conda.sh  # adjust if different
conda activate zero
pip install omegaconf 

cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code/

# --- Diagnostic Info ---
echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi
echo "============================="

echo "=== CUDA Toolkit Info (nvcc) ==="
nvcc --version || echo "nvcc not found"
echo "==============================="

echo "=== PyTorch CUDA Info ==="
python - <<'PYCODE'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected", torch.cuda.device_count(), "GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
        print("   Compute capability:", torch.cuda.get_device_capability(i))
        print("   Memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2))
print("PyTorch CUDA build version:", torch.version.cuda)
PYCODE
echo "==============================="

# --- Run your actual script ---
echo "Starting main script: scripts/train/pubmed_32.sh"
sh scripts/train/pubmed_32.sh
echo "Finished main script."

echo "=========================================="
echo " Finished Eval PubMed Job at: $(date)"
echo "=========================================="
