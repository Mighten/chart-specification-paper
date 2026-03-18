"""
构造 Chart2code SFT 消融实验数据集
"""

import argparse
import json
import os

from pathlib import Path


system_prompt_CHART2CODE_raw_chartmimic = f"You are an expert Python developer specializing in writing Matplotlib code based on a given image. I found a very nice image in a STEM paper, but there is no corresponding source code available. I need your help to generate Python code that can reproduce the image based on the picture I provide. Additionally, I will not provide you with the actual data in the image, so you have to extract the actual data by yourself and based on the extracted data to reproduce the image. Ensure that the code you provide can be executed directly without requiring me to add additional variables.\n\nNow, please provide me with the Matplotlib code that reproduces the image below.\n"


def build_alpaca_sft_item_dict(item_img_filepath_list, gt_python_code):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt_CHART2CODE_raw_chartmimic,
            },
            {
                "role": "user",
                "content": "<image>",
            },
            {
                "role": "assistant",
                "content": f"```python\n{gt_python_code}\n```",
            },
        ],
        "images": item_img_filepath_list,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default="/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/chart2code_combo_train_3k_20251030_1732.jsonl")
    parser.add_argument('--train_img_base_path', default='/home/MY_USERNAME/datasets/mingchen_chart2code_combo_train_3k_20251030')
    parser.add_argument('--out_alpaca_path', default='/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/chart2code_combo_train_3k_alpaca_sft.json')
    args = parser.parse_args()


    res_dict_list = []
    with open(args.train_data_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                img_filepath = Path(os.path.join(args.train_img_base_path, row_dict["img"])).resolve().absolute().as_posix()
                item_img_filepath_list = [img_filepath]
                gt_python_code = row_dict["code"]
                alpaca_item_dict = build_alpaca_sft_item_dict(item_img_filepath_list, gt_python_code)
                res_dict_list.append(alpaca_item_dict)
            except Exception as e:
                print(e)

    with open(args.out_alpaca_path, "w", encoding="utf-8") as f_out:
        json.dump(res_dict_list, fp=f_out, ensure_ascii=False, indent=4)
