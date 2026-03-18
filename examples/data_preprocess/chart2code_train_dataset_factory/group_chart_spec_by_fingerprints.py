import collections
import json



input_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/备份Parquet1：Chart2code全量IR抽取1104之前最好版本结果20251222_1621.jsonl"


chart_type_set = set()
panel_count_set = set()
panel_layout_set = set()

chart_panel_layout_dict = collections.defaultdict(set)
chart_has_values_dict = collections.defaultdict(bool)
chart_has_function_dict = collections.defaultdict(bool)






if __name__ == "__main__":
    with open(input_jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                spec_dict = row_dict["spec"]
                if not spec_dict["chart_type"]:
                    continue
                chart_type_str = spec_dict["chart_type"]
                chart_type_set.add(chart_type_str)
                if spec_dict["panel_count"] > 9:
                    continue
                panel_count_set.add(spec_dict["panel_count"])
                if not spec_dict["panel_layout"] or spec_dict["panel_layout"][0] * spec_dict["panel_layout"][1] > 9:
                    continue
                panel_layout_tuple = tuple(spec_dict["panel_layout"])
                panel_layout_set.add(panel_layout_tuple)

                chart_panel_layout_dict[chart_type_str].add(panel_layout_tuple)
            except Exception as e:
                print(e)


    print("len(chart_type_set) = ", len(chart_type_set), ":", chart_type_set)
    print("------")
    print("len(panel_count_set) = ", len(panel_count_set), ":", panel_count_set)
    print("------")
    print("len(panel_layout_set) = ", len(panel_layout_set), ":", panel_layout_set)
    print("------")


    ans = 0
    for chart_type, layout_set in chart_panel_layout_dict.items():
        print(chart_type, "--->", layout_set)
        ans += len(layout_set)
    
    print("------")
    print("total:", ans)
