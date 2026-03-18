""" Preprocess dataset for knights and knaves logic task """

import io
import base64
import json
import os
import sys
import time
from datasets import Dataset, load_dataset
from datasets.arrow_dataset import Dataset as parquet_dataset_type
from datetime import datetime

from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from random import shuffle
from PIL import Image

system_prompt_chart_qa_cot_v1 = 'You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'


# 解析打包好的 Parquet Dataset，取出 prompt 字段，计算最大 Prompt 长度，精确控制 veRL 参数，防止显存爆炸
def get_parquet_dataset_max_prompt_len(parquet_dataset: parquet_dataset_type):
    max_prompt_len = 0
    for item in parquet_dataset:
        max_prompt_len = max(max_prompt_len, len(item['prompt'][0]['content']))
    return max_prompt_len


# 伪生成器，仿照 JSONL 方式，遍历JSON 数组文件，逐个生成 Prompt 所需的元素
def gen_from_ChartQA_json(json_array_filepath):
    with open(json_array_filepath, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
        for obj in json_list:
            yield obj


def get_image_base64_data_url(img_filepath: str) -> str:
        # 打开图像（支持 jpg、webp 等格式）
    with Image.open(img_filepath) as img_obj, io.BytesIO() as buffer:
        img_obj = img_obj.convert('RGB')
        img_obj.save(buffer, format="PNG")
        buffer.seek(0)
        # base64 编码
        encoded_str = base64.b64encode(buffer.read()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded_str}"
        return data_url


def build_apache_parquet_for_mllm_grpo():
    parser = argparse.ArgumentParser()
    # 当前路径： examples/data_preprocess/verl_grpo_mllm_chartqa_human_dataset.py
    parser.add_argument('--local_dir', default=f'/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/chartqa_mllm-{datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_path', default='/home/MY_USERNAME/dmc61-workspace/chart-benchmarks/ChartQA/ChartQA Dataset/train/train_human.json')
    parser.add_argument('--train_img_base_path', default='/home/MY_USERNAME/dmc61-workspace/chart-benchmarks/ChartQA/ChartQA Dataset/train/png')
    parser.add_argument('--train_size', type=int, default=3008)
    parser.add_argument('--test_data_path', default='/home/MY_USERNAME/dmc61-workspace/chart-benchmarks/ChartQA/ChartQA Dataset/val/val_human.json')
    parser.add_argument('--test_img_base_path', default='/home/MY_USERNAME/dmc61-workspace/chart-benchmarks/ChartQA/ChartQA Dataset/val/png')
    parser.add_argument('--test_size', type=int, default=304)
    parser.add_argument('--data_source', type=str, help='Parquet data_source 字段', default='chartqa_human_label')
    parser.add_argument('--ability', type=str, help='Parquet ability 字段', default='chart_qa')
    
    args = parser.parse_args()
    parquet_data_source = args.data_source
    parquet_ability = args.ability
    local_dir = args.local_dir

    raw_train_dataset = Dataset.from_generator(gen_from_ChartQA_json, gen_kwargs={'json_array_filepath': args.train_data_path})
    print('the original train dataset has: ', len(raw_train_dataset))

    raw_test_dataset = Dataset.from_generator(gen_from_ChartQA_json, gen_kwargs={'json_array_filepath': args.test_data_path})
    print('the original test dataset has:', len(raw_test_dataset))

    # # 打乱数据集  # 临时测试，暂时不打乱
    # shuffled_train_dataset = raw_train_dataset.shuffle(seed=42)
    # shuffled_test_dataset = raw_test_dataset.shuffle(seed=43)

    # 该方法返回一个 function类型对象（输入为数据集的每一项item_dict和idx），用于并行构建 Parquet
    def make_map_fn(dataset_split_name):
        # 辅助方法，将图像封装成 list of dict
        def build_mllm_image_metadata_for(dataset_split_name, img_filename) -> list:
            img_base_dir_dict = {
                'train': args.train_img_base_path,
                'val': args.test_img_base_path,
            }
            img_base_dir = img_base_dir_dict[dataset_split_name]
            img_filepath = os.path.join(img_base_dir, img_filename)
            return [
                {"image_url": get_image_base64_data_url(img_filepath)}
            ]

        def process_fn(item_dict, idx):
            imgname = item_dict['imgname']
            question = item_dict['query']
            label = item_dict['label']
            # 构造 Parquet 每一条数据的字段
            data = {
                "data_source": parquet_data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt_chart_qa_cot_v1},
                    {"role": "user",   "content": f'<image> {question}'}
                ],
                "ability": parquet_ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": label
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

    # 验证集
    TEST_SIZE = args.test_size
    selected_test_dataset = raw_test_dataset.select(range(TEST_SIZE))
    print('Selected testing dataset:', len(selected_test_dataset))
    test_dataset = selected_test_dataset.map(function=make_map_fn('val'), with_indices=True, desc='Marshal val')

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    
    # timestamp = time.strftime(r"%Y%m%d_%H%M%S", time.localtime())

    train_parquet_filepath = os.path.join(local_dir, f'chartqa_human_train.parquet')
    train_dataset.to_parquet(train_parquet_filepath)
    train_dataset.to_csv(f'{train_parquet_filepath}.csv', index=False)

    test_parquet_filepath = os.path.join(local_dir, f'chartqa_human_val.parquet')
    test_dataset.to_parquet(test_parquet_filepath)
    test_dataset.to_csv(f'{test_parquet_filepath}.csv', index=False)
    print('')
    print(f'Output Training ChartX Apache Parquet dataset filepath:', train_parquet_filepath)
    print(f'Output Testing ChartX Apache Parquet dataset filepath:', test_parquet_filepath)


    # hdfs_dir = args.hdfs_dir
    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)


# 正式构建 Apache Parquet 格式 在 61 服务器构建
if __name__ == '__main__':
    build_apache_parquet_for_mllm_grpo()


# 手工检查验证代码
if __name__ == '__main__：手工检查验证代码':
    import pandas as pd
    df = pd.read_csv('/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/chartqa_mllm-20250627_150437/chartqa_human_val.parquet.csv')
    print(df)
