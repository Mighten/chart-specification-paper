#!/bin/bash

# Huawei 2012 Lab ::Area 61::

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# 【临时调试】用于解决：Watchdog caught collective operation timeout: WorkNCCL(SeqNum=99123, OpType=ALLREDUCE, ...) ran for 1800076 milliseconds before timing out.
export NCCL_TIMEOUT=7200
# 下列内容不激活，
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# 【临时调试】强制 CUDA 操作以同步方式执行，即 CPU 会等待每个 CUDA 操作完成后再继续执行后续代码。
# export CUDA_LAUNCH_BLOCKING=1
# Use py-spy to find out which line is blocked:
#   $ py-spy dump --pid <the pid of your hanged job>

# 禁止在 ProcessPoolExecutor 内部对 reward_tensor 做并发操作，否则会引起:
#   RuntimeError: CUDA error: an illegal memory access was encountered
#   CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

# 61 服务器训练基准模型地址
# 注意：qwen模型多模态处理需要提前安装 qwen_vl_utils 依赖： $ pip install qwen_vl_utils
MODEL_PATH=/data/MY_USERNAME/hf_dl/Qwen2.5-VL-7B-Instruct

CURRENT_DATE_TIME=$(date  +"%Y%m%d_%H%M%S").log
CURRENT_LOGGING_FILE_NAME="verl_grpo_hybrid_mllm_$CURRENT_DATE_TIME"

# how to switch on mode.fsdp_config.cpu_offload=True ?
    # +actor_rollout_ref.model.fsdp_config.cpu_offload=True \ ????????????????
    # +actor_rollout_ref.model.fsdp_config.offload_params=True \ ????????????????

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/hybrid_mllm-20250829_123253/hybrid_train.parquet \
    data.val_files=/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/hybrid_mllm-20250829_123253/hybrid_train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=2000 \
    data.max_response_length=3000 \
    data.truncation=left \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=48 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.limit_images=1 \
    +actor_rollout_ref.rollout.limit_videos=0 \
    +actor_rollout_ref.rollout.limit_audios=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.max_model_len=5000 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=48 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='hybrid_mllm' \
    trainer.experiment_name='chart2code_and_ChartQA' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=targets/hybrid_mllm_qwen2.5_vl_7b_instruct \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.total_epochs=2 \
    "$@" 2>&1 | tee -a $CURRENT_LOGGING_FILE_NAME

# 1. 快速后台启动
#    ray stop --force
#    pyclean . && nohup bash hybrid-mllm-grpo.sh > /dev/null 2>&1 & 
# 
# 2. 一键后台 kill (发送 SIGTERM)
#    kill   $(ps aux | grep hybrid_mllm_qwen2.5_vl_7b_instruct | grep -Ev 'grep --color=auto' | awk '{print $2}' | tr '\n' ' ')
#
# 3. 取出最新日志文件中的 Ray Dashboard 地址打印在控制台，按住 Ctrl 键点击拼装好的 URL地址，借助 VSCode Remote Server 建立 SSH 隧道
#     echo "Ray Dashboard via VSCode SSH Tunnel(Ctrl + Click):  http://$(head -5 $(ls verl_*.log | sort | tail -n 1) | grep --color=never -Po 'View the dashboard at \K.*')"
#
# 4. 快速查看当前进度
#      echo "Latest training progress: $(tail -2000 $(ls verl_*.log | sort | tail -n 1) | grep -Po '\K END global step ([0-9]+/[0-9]+)' | sort | tail -1)"
# process reward 
