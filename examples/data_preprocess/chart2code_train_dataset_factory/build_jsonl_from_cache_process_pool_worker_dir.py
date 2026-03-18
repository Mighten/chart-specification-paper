import os
import json

# input_json_base_dir = "/home/MY_USERNAME/datasets/ChartCoder/Chart2Code-160k/cache_IR_full_chart2code-160k_20251212"
# output_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：Chart2code全量IR抽取1104之前最好版本结果20251212_1640.jsonl"

# input_json_base_dir = "/home/MY_USERNAME/datasets/ChartCoder/Chart2Code-160k/cache_IR_full_chart2code-160k_20251218"
# output_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：Chart2code全量IR抽取1104之前最好版本结果20251222_1621.jsonl"

input_json_base_dir = "/home/MY_USERNAME/datasets/ReachQA/cache_pool_worker_dir_reachqa_20251222"
output_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet3：ReachQA-train全量IR抽取1104之前最好版本结果20251222_1634.jsonl"



if __name__ == '__main__':
    with open(output_jsonl_filepath, "w", encoding="utf-8") as f_out:
        for filename in os.listdir(input_json_base_dir):
            json_filepath = os.path.join(input_json_base_dir, filename)
            with open(json_filepath, "r", encoding="utf-8") as f:
                try:
                    row_dict = json.load(f)
                    f_out.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(e)

