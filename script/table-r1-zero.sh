#!/usr/bin/env bash
set -x

project_name='Table-R1'
exp_name='Table-R1-Zero-7B'
home_dir='Table-R1'
mkdir -p "$home_dir/log/$project_name/$exp_name"
save_path="$home_dir/checkpoints/$project_name/$exp_name"
mkdir -p $save_path

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE="$home_dir/data/table-r1-zero-dataset_train.parquet"
TEST_FILE="$home_dir/data/table-r1-zero-dataset_test.parquet"

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 1))

train_bsz=256
mini_bsz=256

rollout_n=16
train_temperature=1.0
val_temperature=0.6

clip_ratio_low=0.2
clip_ratio_high=0.28

ppo_max_token_len_per_gpu=$((1024 * 16))
log_prob_max_token_len_per_gpu=$((1024 * 32))
max_num_batched_tokens=$((1024 * 32))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=2 $@ 2>&1 | tee >(split -b 5M -d --additional-suffix=.log - "log/$project_name/$exp_name/")
