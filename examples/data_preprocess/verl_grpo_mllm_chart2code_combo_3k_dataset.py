""" Preprocess dataset for chart2code task """

import io
import base64
import json
import os
import sys
import time
import numpy as np

from datasets import Dataset, load_dataset
from datasets.arrow_dataset import Dataset as parquet_dataset_type
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from random import shuffle
from PIL import Image

rl_system_prompt_chart2code_cot_v1 = """You are an expert Python developer specializing in generating matplotlib code to reproduce a given chart. Please think through the reasoning process in your mind and then provides the user with the matplotlib code that can reproduce the picture. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think>
reasoning process here
</think>
<answer>
```python
matplotlib code here
```
</answer>
"""


# 伪生成器，仿照 JSONL 方式，遍历JSON 数组文件，逐个生成 Prompt 所需的元素
def gen_from_chart2code_json(jsonl_filepath):
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                for dummy_key_name in ('chartmimic', 'major_chart_type', 'minor_chart_type', 'level'):
                    # 移除无关字段
                    if dummy_key_name in row_dict:
                        row_dict.pop(dummy_key_name)

                if not row_dict['spec'] or not row_dict['spec']["chart_type"]:
                    continue

                # spec 字典变为 JSON 序列化字符串
                row_dict['spec'] = json.dumps(row_dict['spec'], ensure_ascii=False)
                yield row_dict
            except Exception as e:
                print("Failed to read JSONL Line", e)
                continue


def get_image_base64_data_url(img_filepath: str, max_resolution: int = 672, patch_size: int = 14) -> str:
    """
    将指定路径的图像文件转换为 Base64 编码的 Data URL。

    把图片抗锯齿缩放提前到 Parquet 生成阶段，进一步提高分辨率上限，而不是在 Rollout 阶段逐次 resize

    参数:
        img_filepath (str): 图像文件的路径。
        max_resolution (int, 可选): 图像的最大分辨率（宽度或高度），默认为 672
        patch_size (int, 可选): 用于调整图像缩放比例的块大小，通常用于模型输入对齐， 默认为 14。

    返回:
        str: Base64 编码的 Data URL 字符串，格式为 'data:image/png;base64,...'
    """
    def find_best_resize(original_size, scale_resolution, patch_size):
        def ensure_divide(length, patch_size):
            return max(int(np.floor(length / patch_size) * patch_size), patch_size)
        width, height = original_size
        # 输入图片的长或宽的像素数都必须在 max_resolution 范围以内
        if max(height, width) > scale_resolution:
            scale = scale_resolution / max(width, height)
            width = width * scale
            height = height * scale
        best_width = ensure_divide(width, patch_size)
        best_height = ensure_divide(height, patch_size)
        return (best_width, best_height)

    with Image.open(img_filepath) as img_obj, io.BytesIO() as buffer:
        img_obj = img_obj.convert('RGB')
        best_size = find_best_resize(img_obj.size, max_resolution, patch_size)
        img_obj = img_obj.copy().resize(best_size, Image.Resampling.LANCZOS)
        img_obj.save(buffer, format="PNG")
        buffer.seek(0)
        # base64 编码
        encoded_str = base64.b64encode(buffer.read()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded_str}"
        return data_url


def build_apache_parquet_for_mllm_grpo():
    parser = argparse.ArgumentParser()
    # 当前路径： examples/data_preprocess/verl_grpo_mllm_chart2code_combo_3k_dataset.py
    parser.add_argument('--local_dir', default=f'data-rl-parquet/chart2code_mllm-{datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_path', default="/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/chart2code_combo_train_3k_20251030_1732.jsonl")
    parser.add_argument('--train_img_base_path', default='/home/MY_USERNAME/datasets/mingchen_chart2code_combo_train_3k_20251030')
    parser.add_argument('--train_size', type=int, default=3008)
    parser.add_argument('--data_source', type=str, help='Parquet data_source 字段', default='chart2code_combo_3k')
    parser.add_argument('--ability', type=str, help='Parquet ability 字段', default='chart2code')
    
    args = parser.parse_args()
    parquet_data_source = args.data_source
    parquet_ability = args.ability
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)


    raw_train_dataset = Dataset.from_generator(gen_from_chart2code_json, gen_kwargs={'jsonl_filepath': args.train_data_path})
    print('the original train dataset has: ', len(raw_train_dataset))

    # 打乱数据集
    raw_train_dataset = raw_train_dataset.shuffle(seed=42)
    # shuffled_test_dataset = raw_test_dataset.shuffle(seed=43)

    # 该方法返回一个 function类型对象（输入为数据集的每一项item_dict和idx），用于并行构建 Parquet
    def make_map_fn(dataset_split_name):
        # 辅助方法，将图像封装成 list of dict
        def build_mllm_image_metadata_for(dataset_split_name, img_filename) -> list:
            img_filepath = os.path.join(args.train_img_base_path, img_filename)
            return [
                {"image_url": get_image_base64_data_url(img_filepath)}
            ]

        def process_fn(item_dict, idx):
            imgname = item_dict['img']
            spec = item_dict['spec']
            # 构造 Parquet 每一条数据的字段
            data = {
                "data_source": parquet_data_source,
                "prompt": [
                    {"role": "system", "content": rl_system_prompt_chart2code_cot_v1},
                    {"role": "user",   "content": f'<image> Please generate matplotlib code according the given chart image.'}
                ],
                "ability": parquet_ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": spec,
                },
                "extra_info": {
                    "id": item_dict['id'],
                    "split": dataset_split_name,
                    "spec": spec,
                },
                "images": build_mllm_image_metadata_for(dataset_split_name, imgname),
            }
            return data
        return process_fn

    # 随机抽样
    # random_selected_train_dataset = shuffled_train_dataset.select(range(TRAIN_SIZE))
    # print('randomly selected training dataset:', len(random_selected_train_dataset))
    # train_dataset = random_selected_train_dataset.map(function=make_map_fn('train'), with_indices=True)

    # 训练集
    TRAIN_SIZE = args.train_size
    selected_train_dataset = raw_train_dataset.select(range(TRAIN_SIZE))
    print('Selected training dataset:', len(selected_train_dataset))
    train_dataset = selected_train_dataset.map(function=make_map_fn('train'), with_indices=True, desc='Marshal train')

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    
    train_parquet_filepath = Path(os.path.join(local_dir, f'chart2code_combo_3k_train.parquet')).resolve().absolute().as_posix()
    train_dataset.to_parquet(train_parquet_filepath)

    print('')
    print(f'Output Training chart2code-combo-3k Apache Parquet dataset filepath:', train_parquet_filepath)
    print('')

    train_dataset.to_csv(f'{train_parquet_filepath}.csv', index=False)


# 正式构建 Apache Parquet 格式 在 61 服务器构建
# conda activate svgr1
# python examples/data_preprocess/verl_grpo_mllm_chart2code_combo_3k_dataset.py
if __name__ == '__main__':
    build_apache_parquet_for_mllm_grpo()
