# -*- coding: utf-8 -*-
import collections
import json
import random

input_jsonl_filepath = "./备份Parquet3：ReachQA待筛选数据集3187个20251030_0915.jsonl"
output_jsonl_filepath = "./备份Parquet3：最终ReachQA待筛选数据集1365个20251030_1534.jsonl"

input_reach_qa_json_filepath = './备份Parquet3：ReachQA待筛选数据集3187个20251030_0915_chart_type_to_case_id_name_list.json'
output_reach_qa_json_filepath = './备份Parquet3：ReachQA筛选后数据集1365个20251030_1400_chart_type_to_case_id_name_list.json'

quota_dict = {
    "mix": 483,
    "bar": 66,
    "line": 116,
    "multi_axes": 153,
    "3d": 141,
    "boxplot": 57,
    "scatter": 12,
    "error": 101,
    "pie": 18,
    "treemap": 28,
    "graph": 78,
    "radar": 59,
    "ring": 4,
    "heatmap": 32,
    "density": 1,
    "quiver": 0,
    "violin": 1,
    "contour": 0,
    "histogram": 10,
    "rose": 6,
}


if __name__ == '__main__':
    type_to_id_str_list_dict = None
    with open(input_reach_qa_json_filepath, 'r', encoding='utf-8') as f:
        type_to_id_str_list_dict = json.load(f)

    selected_case_type_dict = collections.defaultdict(list)
    random.seed(42)

    # 随机选出 对应配额的 case_id
    for type_name, id_str_list in type_to_id_str_list_dict.items():
        selected_case_type_dict[type_name] = random.sample(id_str_list, quota_dict[type_name])

    with open(output_reach_qa_json_filepath, 'w', encoding='utf-8') as f:
        json.dump(selected_case_type_dict, f, ensure_ascii=False, indent=4)

    # 构建 case_id_name_set
    case_id_name_set = set()
    for type_name, case_id_name_list in selected_case_type_dict.items():
        # print(type_name, len(case_id_name_list))
        for case_id_str in case_id_name_list:
            case_id_name_set.add(case_id_str)

    print(f"筛选后总数据集大小：{len(case_id_name_set)}")

    with open(output_jsonl_filepath, 'w', encoding='utf-8') as f_out, open(input_jsonl_filepath, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                case_id_str = row_dict['id']
                if case_id_str in case_id_name_set:
                    f_out.write(line + "\n")
            except Exception as e:
                print(e)
