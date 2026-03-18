
import json
import re
import time

from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

from . import compute_score
from .evaluate_spec import get_spec_from_matplotlib_code

output_case_id_to_llm_ir_dict_json_filepath = r"D:\veRL-MLLM-GRPO\case_id_to_llm_ir_dict.json"
output_mock_input_jsonl_filepath = r"D:\veRL-MLLM-GRPO\mock.jsonl"


def unpack_python_code_fence(text: str) -> str:
    matches = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    return matches[0].strip() if matches else ''


def worker_func_parse_llm_ir(arg_dict: dict) -> dict:
    case_id_str = arg_dict["case_id"]
    raw_answer_str = arg_dict["raw_answer"]
    code_str = unpack_python_code_fence(raw_answer_str)
    return {case_id_str: get_spec_from_matplotlib_code(code_str)}


# 将任务提交到进程池，防止GIL
def batch_parse_llm_ir_process_pool(max_workers=64, global_working_func=None, args_list=None):
    case_id_to_llm_ir_dict = dict()
    with ThreadPoolExecutor(max_workers) as executor, tqdm(total=len(args_list), desc='Batch Parse LLM IR') as pbar:
        futures_list = [executor.submit(global_working_func, arg) for arg in args_list]
        for first_completed_future in as_completed(futures_list):
            try:
                result = first_completed_future.result()
                case_id_to_llm_ir_dict.update(result)
            except Exception as e:
                print(e)
            finally:
                pbar.update(1)
    return case_id_to_llm_ir_dict


if __name__ == "__main__:提前提取 rollout LLM IR":
    added_case_id_name_set = set()
    args_list = []
    with open(output_mock_input_jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                if row_dict["id"] in added_case_id_name_set:
                    continue
                added_case_id_name_set.add(row_dict["id"])
                args_list.append({
                    "case_id": row_dict["id"],
                    "raw_answer": row_dict["raw_answer"],
                })
            except Exception as e:
                print(e)

    case_id_to_llm_ir_dict = batch_parse_llm_ir_process_pool(global_working_func=worker_func_parse_llm_ir, args_list=args_list)
    with open(output_case_id_to_llm_ir_dict_json_filepath, 'w', encoding='utf-8') as f_json:
        json.dump(case_id_to_llm_ir_dict, fp=f_json, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    visited_case_id_name_set = set()
    with open(output_mock_input_jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                raw_answer_str = row_dict["raw_answer"]
                case_id_str = row_dict["id"]
                if case_id_str in visited_case_id_name_set:
                    continue
                visited_case_id_name_set.add(case_id_str)
                ground_truth_str = json.dumps(row_dict["gt"], ensure_ascii=False)
                extra_info_dict = {
                    "id": case_id_str,
                    "split": "train",
                }
                inject_code_ir_dict = row_dict["code_ir"]
            except Exception as e:
                print(e)

            begin_time = time.time()
            reward_score = compute_score(raw_answer_str, ground_truth_str, extra_info_dict, inject_code_ir_dict)
            end_time = time.time()
            print(f"Case ID {case_id_str} final reward score = {reward_score}, cost {end_time - begin_time}")
