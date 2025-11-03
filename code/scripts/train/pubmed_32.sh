# export CUDA_VISIBLE_DEVICES=4,7

DATE=$(date '+%Y-%m-%d-%H-%M-%S')

python3 -m verl.trainer.main_ppo \
    data.train_files=/projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code/data/search_engine/pubmed_32/train.parquet \
    data.val_files=/projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code/data/search_engine/pubmed_32/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=500 \
    data.max_response_length=500 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    critic.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=500 \
    trainer.project_name=pubmed_search \
    trainer.experiment_name=pubmed_search_32 \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    critic.model.path=meta-llama/Llama-3.2-3B-Instruct \
    trainer.total_epochs=10 2>&1 | tee exp_log/32-ppo-verl_demo_$DATE.log \
    
