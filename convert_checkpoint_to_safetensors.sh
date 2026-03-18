#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS

VERL_CHECKPOINT_DIR=/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/targets/chart2code_mllm_qwen2.5_vl_7b_instruct_ablation_scaling_ChartStruct_2k_select_charts/global_step_315

# Checkpoints
# If you want to convert the model checkpoint into huggingface safetensor format, please refer to scripts/model_merger.py.
#  https://verl.readthedocs.io/en/latest/faq/faq.html#checkpoints
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  $VERL_CHECKPOINT_DIR/actor \
    --target_dir $VERL_CHECKPOINT_DIR/merged_hf_model


# conda activate svgr1
# bash convert_checkpoint_to_safetensors.sh
