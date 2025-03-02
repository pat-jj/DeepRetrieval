# Panacea-R1

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Get started

**Data Preparation**
```
conda activate zero
python examples/data_preprocess/patient_trial_matching.py
```

**Reward function**
```
code/verl/utils/reward_score
```

**Modify the compute_score_fn in code/verl/trainer/main_ppo.py**

**Monitor info**
```
code/verl/trainer/ppo/ray_trainer.py - compute_reward_metrics()
```

### Run Training
```
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script


**3B+ model**
```
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=data/matching/qwen-instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=matching-qwen2.5-3b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="[Your_key]"
export HF_HOME="/srv/local/data/linjc/hub"

export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train_tiny_zero_ppo.sh
```