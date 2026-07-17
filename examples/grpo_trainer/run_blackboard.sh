set -x
ENGINE=${1:-vllm}
#export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=4
val_data_size=50
group_size=8
dataset_name='frames'

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.blackboard \
    --dataset_name $dataset_name \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    data.trust_remote_code=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    trainer.default_local_dir=$HOME/autodl-tmp/verl_agent_blackboard/grpo_qwen3-4b-instruct-frames-ds32-highlr-tool \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.lora_rank=16 \
    data.train_files=$HOME/verl-agent/data/$dataset_name/train.parquet \
    data.val_files=$HOME/verl-agent/data/$dataset_name/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=28000 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct \
    actor_rollout_ref.model.path=$HOME/autodl-tmp/Qwen3-4B-Instruct \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Blackboard \
    env.seed=0 \
    env.max_steps=8 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_blackboard' \
    trainer.experiment_name='grpo_qwen3-4b-instruct-frames-ds32-highlr-tool' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True $@
