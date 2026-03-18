# -*- coding: utf-8 -*-

import json
import os
import re


pattern = r"\[train-(.*?)\]"


log_input_filepath = "./verl_grpo_chart2code_mllm_20250930_130726.log"
input_jsonl_data_parquet_filepath = "data-rl-parquet/raw_data_jsonl/备份：VL提取IR与 ChartMimic lowlevel筛选后合并数据集-20250925_0934.jsonl"
image_base_dir = r"D:\datasets\Mingchen_Chart_Reasearch_chart2code_20250711_1340\chart2code-160k-base_dir\images"

mock_rollout_jsonl_filepath = "./mock.jsonl"

output_valid_jsonl_data_parquet_filepath = "data-rl-parquet/raw_data_jsonl/手动筛选chart2code160k数据集1587个_20251009_0923.jsonl"


valid_case_id_to_code_ir_dict = dict()


def get_valid_case_id_set():
    with open(log_input_filepath, "r", encoding='utf-8') as f:
        for line in f:
            if not line or "(PASSED format + compile)" not in line:
                continue
            case_id = re.search(pattern, line).group(1)
            code_ir_str = line.split("with code IR extracted: ", 1)[1]
            valid_case_id_to_code_ir_dict[case_id] = json.loads(code_ir_str)
    print(f"yield {len(valid_case_id_to_code_ir_dict)} valid case ID item(s).")
    return valid_case_id_to_code_ir_dict


def get_valid_case_id_name_set_from_jsonl_and_image_dir():
    """
    获取 image case id 和 jsonl case id 交集
    """
    valid_case_id_name_set_from_image_dir = set()
    for filename in os.listdir(image_base_dir):
        if not filename.endswith(".png"):
            continue
        file_base_name = filename.split(".")[0]
        case_id_str = f"chart2code-160k-{file_base_name}"
        valid_case_id_name_set_from_image_dir.add(case_id_str)
    print("from image dir:", len(valid_case_id_name_set_from_image_dir))
    valid_case_id_name_set_from_jsonl = set()
    with open(input_jsonl_data_parquet_filepath, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                case_id_str = row_dict["id"]
                valid_case_id_name_set_from_jsonl.add(case_id_str)
            except Exception as e:
                print(e)
    print("from jsonl:", len(valid_case_id_name_set_from_jsonl))
    final_valid_case_id_name_set = valid_case_id_name_set_from_jsonl & valid_case_id_name_set_from_image_dir
    print("final valid case id set:", len(final_valid_case_id_name_set))
    return final_valid_case_id_name_set


if __name__ == '__main__:重新组装 dataset Parquet RL JSONL':
    valid_case_id_name_set = get_valid_case_id_name_set_from_jsonl_and_image_dir()
    with open(input_jsonl_data_parquet_filepath, "r", encoding='utf-8') as f_in, open(output_valid_jsonl_data_parquet_filepath, "w", encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                case_id_str = row_dict["id"]
                if case_id_str not in valid_case_id_name_set:
                    continue
                f_out.write(line + "\n")
            except Exception as e:
                print(e)


if __name__ == '__main__':
    num_rollouts = 0
    valid_case_id_name_set = get_valid_case_id_name_set_from_jsonl_and_image_dir()
    valid_case_id_to_code_ir_dict = get_valid_case_id_set()

    with open(log_input_filepath, "r", encoding='utf-8') as f_in, open(mock_rollout_jsonl_filepath, "w", encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line or "---sampling-rollouts---" not in line:
                continue
            case_id = re.search(pattern, line).group(1)
            if case_id not in valid_case_id_to_code_ir_dict or case_id not in valid_case_id_name_set:
                continue
            line = line.split("[ROLLOUT] ", 1)[1]
            try:
                row_dict = json.loads(line)
                row_dict["id"] = case_id
                row_dict["code_ir"] = valid_case_id_to_code_ir_dict[case_id]
                f_out.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
                num_rollouts += 1
            except Exception as e:
                print(e)
    print("valid samples:", num_rollouts)
