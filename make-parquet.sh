#!/bin/bash

# conda activate svgr1

# 构建 ChartQA Parquet
# python examples/data_preprocess/verl_grpo_mllm_chartqa_human_dataset.py

# 构建 chart2code Parquet
python examples/data_preprocess/verl_grpo_mllm_chart2code_combo_3k_dataset.py
