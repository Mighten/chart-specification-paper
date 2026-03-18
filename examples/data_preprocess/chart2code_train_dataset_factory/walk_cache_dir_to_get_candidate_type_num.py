import collections
import json
import os


# worker_cache_dir_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：Chart2code全量IR抽取1104之前最好版本结果20251222_1621.jsonl"

worker_cache_dir_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet3：ReachQA-train全量IR抽取1104之前最好版本结果20251222_1634.jsonl"


if __name__ == "__main__：从 cache dir 中解析":
    cache_base_dir = "/home/MY_USERNAME/datasets/ReachQA/cache_pool_worker_dir_reachqa_20251222"
    type_to_case_id_list = collections.defaultdict(list)
    for filename in os.listdir(cache_base_dir):
        if not filename.endswith(".json"):
            continue
        try:
            json_filepath = os.path.join(cache_base_dir, filename)
            with open(json_filepath, "r", encoding="utf-8") as f:
                row_dict = json.load(f)
                id_str = row_dict["id"]
                code_str = row_dict["code"]
                spec_dict = row_dict["spec"]
                chart_type_str = spec_dict["chart_type"]
                type_to_case_id_list[chart_type_str].append(id_str)
        except Exception as e:
            print(e)

    print("chart_type, chart_num")
    for chart_type, case_id_list in type_to_case_id_list.items():
        print(repr(chart_type), ", ", len(case_id_list))




if __name__ == "__main__":
    type_to_case_id_list = collections.defaultdict(list)
    with open(worker_cache_dir_jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                id_str = row_dict["id"]
                code_str = row_dict["code"]
                spec_dict = row_dict["spec"]
                chart_type_str = spec_dict["chart_type"]
                type_to_case_id_list[chart_type_str].append(id_str)
            except Exception as e:
                print(e)

    print("chart_type, chart_num")
    for chart_type, case_id_list in type_to_case_id_list.items():
        print(repr(chart_type), ", ", len(case_id_list))
