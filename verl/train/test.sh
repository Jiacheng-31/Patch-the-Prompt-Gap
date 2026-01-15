#!/usr/bin/env bash
set -x   # 打印每一行执行的命令，方便调试
# nohup bash test.sh > test1.log 2>&1 & echo $! > test.pid

########################################
# 0. 使用 0–7 号 GPU
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

########################################
# 1. 数据路径：只用你预处理好的 GSM8K
########################################

train_path=/workspace/verl/train/numinamath/train.parquet # 训练集
test_path=/workspace/verl/train/numinamath/test.parquet   # 测试/验证集

MODEL_PATH=/workspace/models/Mistral-7B-Instruct-v0.3

train_files="['$train_path']"
test_files="['$test_path']"

########################################
# 2. LoRA + GRPO 训练
########################################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=504 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=252 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    custom_reward_function.path=/workspace/verl/train/my_reward.py \
    custom_reward_function.name=compute_rewrite_reward \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_rewriting_Mistral7b_lora' \
    trainer.experiment_name='Mistral7b_rewriting_grpo_lora' \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=100 \
    trainer.total_epochs=2 "$@"
