import pandas as pd
from pathlib import Path
import json


input_parquet_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/chart2code_mllm-20260107_175738/chart2code_combo_ChartStruct_4k_train.parquet"


out_sft_dataset_json_filename = Path(input_parquet_filepath).stem + "-Full-SFT.json"
out_sft_dataset_json_filepath = (Path(input_parquet_filepath).parent / Path(out_sft_dataset_json_filename)).resolve().absolute().as_posix()

img_base_dir = "/home/MY_USERNAME/datasets/mingchen_chart2code_combo_train_3k_20251030"


if __name__ == "__main__":
    ans_dict_list = []
    df = pd.read_parquet(input_parquet_filepath)
    for index, row_dict in df.iterrows():
        message_list = []
        message_list.append({
            "role": "system",
            "content": "You are an expert Python developer specializing in writing Matplotlib code based on a given image. I found a very nice image in a STEM paper, but there is no corresponding source code available. I need your help to generate Python code that can reproduce the image based on the picture I provide. Additionally, I will not provide you with the actual data in the image, so you have to extract the actual data by yourself and based on the extracted data to reproduce the image. Ensure that the code you provide can be executed directly without requiring me to add additional variables.\n\nNow, please provide me with the Matplotlib code that reproduces the image below.\n",
        })
        message_list.append({
            "role": "user",
            "content": "<image>",
        })
        message_list.append({
            "role": "assistant",
            "content": f"```python\n{row_dict['code']}\n```"
        })
        ans_dict_list.append({
            "messages": message_list,
            "images": [
                (Path(img_base_dir) / row_dict["img"]).resolve().absolute().as_posix()
            ]
        })

    with open(out_sft_dataset_json_filepath, "w", encoding="utf-8") as f_out:
        json.dump(ans_dict_list, f_out, ensure_ascii=False, indent=4)

    print(f"written {len(ans_dict_list)} obj into JSON: {out_sft_dataset_json_filepath}")

# conda activate svgr1
# python examples/data_preprocess/verl_grpo_convert_parquet_to_SFT_dataset.py
