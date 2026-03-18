import json
import collections


output_chart2code_combo_train_3k_dataset_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/chart2code_combo_train_3k_20251030_1732.jsonl"

jsonl_filepath_list = [
    "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：新IR手动筛选数据集1582个chart2code160k合并20251024_1711.jsonl",
    "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet2：新IR手合成数据集20251030_1149.jsonl",
    "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet3：最终ReachQA待筛选数据集1365个20251030_1534.jsonl",
]


if __name__ == "__main__：之前简单拼接三个 JSONL":
    with open(output_chart2code_combo_train_3k_dataset_jsonl_filepath, "w", encoding="utf-8") as f_out:
        for src_jsonl_filepath in jsonl_filepath_list:
            with open(src_jsonl_filepath, "r", encoding="utf-8") as f_jsonl_src:
                for line in f_jsonl_src:
                    line = line.strip()
                    if not line:
                        continue
                    f_out.write(line + "\n")


already_gathered_chart2code_dataset_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/chart2code_combo_train_3k_20251030_1732.jsonl"
full_chart2code_160k_filtered_dataset_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：Chart2code全量IR抽取1104之前最好版本结果20251212_1640.jsonl"



# 列举剩余可选的 chart2code-160k 数据集中 chart_type -> case_id list
if __name__ == "__main__":
    gathered_item_case_id_name_set = set()

    with open(already_gathered_chart2code_dataset_jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                case_id = row_dict["id"]
                if not case_id.startswith("chart2code-"):
                    continue
                gathered_item_case_id_name_set.add(case_id)
            except Exception as e:
                print(str(e))

    print(f"Gathered chart2code case id in Parquet: ", len(gathered_item_case_id_name_set))

    remaining_candidate_chart_type_to_case_id_list = collections.defaultdict(list)
    with open(full_chart2code_160k_filtered_dataset_jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                spec_dict = row_dict["spec"]
                chart_type = spec_dict["chart_type"]
                case_id = row_dict["id"]
                remaining_candidate_chart_type_to_case_id_list[chart_type].append(case_id)
            except Exception as e:
                print(e)


    for chart_type, case_id_list in remaining_candidate_chart_type_to_case_id_list.items():
        print(chart_type, " ----- ", len(case_id_list))
