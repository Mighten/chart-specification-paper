import os
import json
import shutil
from pathlib import Path

output_jsonl_filepath = './备份Parquet2：新IR手合成数据集20251023_1954.jsonl'

py_base_dir = './result/py'
png_base_dir = './result/png'

keys_name_list = ["id", 'img', 'code', 'spec']


if __name__ == "__main__":
    collected_case_num = 0
    with open(output_jsonl_filepath, "w", encoding="utf-8") as f_out:
        for type_name in os.listdir(py_base_dir):
            cur_type_py_base_dir = os.path.join(py_base_dir, type_name)
            cur_type_png_file_base_dir = os.path.join(png_base_dir, type_name)
            for filename in os.listdir(cur_type_py_base_dir):
                if not filename.endswith('.py'):
                    continue
                py_filepath = os.path.join(cur_type_py_base_dir, filename)
                file_base_name = filename.rsplit('.py', 1)[0]
                case_id_str = f"augment-{type_name}-{file_base_name}"
                src_png_filepath = os.path.join(cur_type_png_file_base_dir, f"{file_base_name}.png")
                target_png_filepath = f"./augment-base_dir/{case_id_str}.png"
                row_dict = {
                    "id": case_id_str,
                    "img": target_png_filepath,
                    "code": "",
                    "spec": dict(),
                }
                with open(py_filepath, "r", encoding="utf-8") as f_py:
                    row_dict["code"] = f_py.read()
                f_out.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
                shutil.copy(src=src_png_filepath, dst=target_png_filepath)
                collected_case_num += 1

    output_jsonl_path = Path(output_jsonl_filepath).absolute().as_posix()
    print(f"Collected {collected_case_num} case(s) and saved to {output_jsonl_path}")
