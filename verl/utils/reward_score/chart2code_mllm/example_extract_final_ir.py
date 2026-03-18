from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import json
import os
import re
import time
import random


from .evaluate_spec import get_spec_from_matplotlib_code
from .lowlevel.code_ir_eval import get_code_ir_dict_from_subprocess_stdout

logger = logging.getLogger(__file__)


def read_jsonl_dict_list(input_jsonl_filepath: str, cache_dir: str) -> list:
    res_dict_list = []
    with open(input_jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                res_dict_list.append({
                    "input_row_dict": row_dict,
                    "cache_result_base_dir": cache_dir,
                })
            except Exception as e:
                print(e)
    return res_dict_list


def submit_to_executor_pool(working_func=None, args_list=None, max_workers=16, process_name: str = 'Processing Chart Specifications'):
    with ProcessPoolExecutor(max_workers) as executor, tqdm(total=len(args_list), desc=process_name) as pbar:
        futures_list = [executor.submit(working_func, arg) for arg in args_list]
        for first_completed_future in as_completed(futures_list):
            pbar.update(1)
            try:
                _ = first_completed_future.result()
            except Exception as e:
                logger.error(f'Thread Pool Worker Error: {e}', exc_info=True)
    return None


def worker_parse_chart_spec(args: dict) -> dict:
    input_row_dict = args["input_row_dict"]
    cache_result_base_dir = args["cache_result_base_dir"]
    case_id = input_row_dict.get("id")
    if not case_id:
        return None
    code_str = input_row_dict["code"]

    final_raw_dict = input_row_dict
    # 缓存命中，直接返回
    cache_filepath = os.path.join(cache_result_base_dir, f"{case_id}.json")
    try:
        if os.path.exists(cache_filepath):
            with open(cache_filepath, "r", encoding="utf-8") as f_cache:
                cached_spec_dict = json.load(f_cache)
                assert len(cached_spec_dict) > 0
                return cached_spec_dict
    except Exception as e:
        logging.warning(f"deprecated cache filepath: {cache_filepath}")
    # 缓存不命中，继续
    last_err_code_ir_dict = None
    last_err_final_ir_dict = None
    try:
        # time.sleep(random.uniform(0, 0.01))
        code_ir_dict = get_code_ir_dict_from_subprocess_stdout(code_str, subprocess_timeout_seconds=30)
        last_err_code_ir_dict = code_ir_dict
        assert len(code_ir_dict) > 0
        ir_dict = get_spec_from_matplotlib_code(code_str)
        last_err_final_ir_dict = ir_dict
        assert len(ir_dict) > 0
        _ = json.dumps(ir_dict, ensure_ascii=False)
        ir_dict["code_ir"] = code_ir_dict
        final_raw_dict["spec"] = ir_dict
        try:
            _ = json.dumps(final_raw_dict, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed test case {case_id} due to {str(e)}, <last_err_code_ir_dict>{repr(last_err_code_ir_dict)}</last_err_code_ir_dict>    <last_err_final_ir_dict>{repr(last_err_final_ir_dict)}</last_err_final_ir_dict>")
            return None

        with open(cache_filepath, "w", encoding="utf-8") as f_cache:
            # 执行结果写入缓存
            json.dump(final_raw_dict, fp=f_cache, ensure_ascii=False, indent=4)
        return final_raw_dict
    except Exception as e:
        logging.error(f"worker failed on case id {case_id} due to {str(e)}")
        return None


def walk_to_get_dict_list_gen() -> dict:
    chart2code_160k_full_json_filepath = "/home/MY_USERNAME/datasets/ChartCoder/Chart2Code-160k/chart2code.json"
    chart2code_160k_img_base_dir = "/home/MY_USERNAME/datasets/ChartCoder/Chart2Code-160k/images"
    """
    遍历 chart2code-160k 原始、全量的数据集，组装成 list of dict，便于统一处理流程
    """
    def unpack_python_codefence(text: str) -> str:
        matches = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
        return matches[0].strip() if matches else ''

    with open(chart2code_160k_full_json_filepath, 'r', encoding='utf-8') as f_json:
        for item_dict in json.load(fp=f_json):
            case_id_str = item_dict["id"]
            img_filepath = item_dict["image"]
            code_str = '' 
            conversation_list = item_dict["conversations"]
            if len(conversation_list) == 2:
                code_str_with_codefence = conversation_list[1]["value"]
                code_str = unpack_python_codefence(code_str_with_codefence)
                if not code_str:
                    continue
                yield {
                    "id": f"chart2code-160k-{case_id_str}",
                    "img": f"./chart2code-160k-base_dir/{img_filepath}",
                    "code": code_str,
                }



# 处理 ChartMimic Direct 600 测试集
if __name__ == '__main__：处理 ChartMimic Direct 600 测试集':
    """
    启动开关3：处理 ChartMimic Direct 600 （测试集）
    """
    chartmimic_direct_600_base_dir = "/home/MY_USERNAME/datasets/ChartMimic/chartmimic-iclr-dataset/direct_600"
    cache_result_base_dir = '/home/MY_USERNAME/datasets/ChartMimic/chartmimic-iclr-dataset/cache_process_pool_direct_600'
    os.makedirs(cache_result_base_dir, exist_ok=True)
    # 准备进程池参数字典列表
    res_raw_data_dict_list = []
    for filename in os.listdir(chartmimic_direct_600_base_dir):
        if not filename.endswith(".py"):
            continue
        file_base_name = filename.replace(".py", "")
        dummy_png_filepath = f"./local_chartmimic_iclr_direct600/{file_base_name}.png"
        case_id_str = f"chartmimic-direct600-{file_base_name}"
        py_filepath = os.path.join(chartmimic_direct_600_base_dir, f"{file_base_name}.py")
        with open(py_filepath, 'r', encoding='utf-8') as f_py:
            res_raw_data_dict_list.append({
            "input_row_dict": {
                "id": case_id_str,
                "img": dummy_png_filepath,
                "code": f_py.read(),
            },
            "cache_result_base_dir": cache_result_base_dir,
        })

    print(f"Gathered {len(res_raw_data_dict_list)} testcase(s) for ChartMimic ICLR Dataset - Direct 600")
    if len(res_raw_data_dict_list) > 0:
        print("Cache Dir:", cache_result_base_dir)
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        print("multiprocessing.set_start_method('spawn', force=True)")
        res_ir_dict_list = submit_to_executor_pool(
                                                   working_func=worker_parse_chart_spec, 
                                                   args_list=res_raw_data_dict_list, 
                                                   max_workers=64, 
                                                   process_name="Parse LLM + Code IR for ChartMimic ICLR Direct 600",
                                                  )
    print("DONE")


# 
if __name__ == '__main__:处理 plot2code':
    plot2code_full_jsonl_filepath = "/home/MY_USERNAME/datasets/Plot2Code/data/python_matplotlib/test/metadata.jsonl"
    plot2code_img_base_dir = "/home/MY_USERNAME/datasets/Plot2Code/data/python_matplotlib/test"

    """
    启动开关4：处理 Plot2code matplotlib test（未测试通过）
    """
    cache_result_base_dir = "/home/MY_USERNAME/datasets/Plot2Code/data/python_matplotlib/cache_process_pool_direct_600"
    os.makedirs(cache_result_base_dir, exist_ok=True)

    res_raw_data_dict_list = []
    with open(plot2code_full_jsonl_filepath, 'r', encoding='utf-8') as f_jsonl:
        for line in f_jsonl:
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = json.loads(line)
                img_name = row_dict["file_name"]
                case_id_str = img_name.rsplit('.', 1)[0].strip()
                dummy_local_plot2code_img_filepath = f"./local_plot2code_img_dir/{img_name}"
                code_str = row_dict["code"]
                res_raw_data_dict_list.append({
                    "input_row_dict": {
                        "id": case_id_str,
                        "img": dummy_local_plot2code_img_filepath,
                        "code": code_str,
                    },
                    "cache_result_base_dir": cache_result_base_dir,
                })
            except Exception as e:
                print(e)

    print(f"Gathered {len(res_raw_data_dict_list)} testcase(s) for Plot2code matplot test")

    if len(res_raw_data_dict_list) > 0:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        print("multiprocessing.set_start_method('spawn', force=True)")
        res_ir_dict_list = submit_to_executor_pool(
                                                   working_func=worker_parse_chart_spec, 
                                                   args_list=res_raw_data_dict_list, 
                                                   max_workers=64, 
                                                   process_name="Parse LLM + Code IR for Plot2Code dataset",
                                                  )
    print("DONE")






# 启动：处理 GRPO RL Parquet 训练集数据（1/2）chart2code-160k 数据集
if __name__ == "__main__：处理 GRPO RL Parquet 训练集数据（1/2）chart2code-160k 数据集":
    """
    启动开关1：处理 chart2code-160k 原始、全量的数据集
    """
    cache_result_base_dir = '/home/MY_USERNAME/datasets/ChartCoder/Chart2Code-160k/cache_IR_full_chart2code-160k_20251218'
    os.makedirs(cache_result_base_dir, exist_ok=True)

    res_raw_data_dict_list = [
        {
            "input_row_dict": x,
            "cache_result_base_dir": cache_result_base_dir,
        }
        for x in walk_to_get_dict_list_gen()]
    res_raw_data_dict_list = res_raw_data_dict_list
    print(f"Gathered {len(res_raw_data_dict_list)} testcase(s).")
    # print(json.dumps(res_raw_data_dict_list[0], ensure_ascii=False, indent=4))

    if len(res_raw_data_dict_list) > 0:
        print("Cache Dir:", cache_result_base_dir)
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        print("multiprocessing.set_start_method('spawn', force=True)")
        res_ir_dict_list = submit_to_executor_pool(working_func=worker_parse_chart_spec, 
                                                   args_list=res_raw_data_dict_list, 
                                                   max_workers=290,
                                                   process_name="Parse chart2code-160k")
    print("DONE")


# 启动：处理 GRPO RL Parquet 训练集数据（2/2）ReachQA Train 数据集
if __name__ == "__main__":
    input_raw_jsonl_filepath = "/home/MY_USERNAME/dmc61-workspace/veRL-MLLM-GRPO/data-rl-parquet/raw_data_jsonl/STALE/备份Parquet3：ReachQA待筛选数据集3211个20251024_1731.jsonl"
    cache_result_base_dir = "/home/MY_USERNAME/datasets/ReachQA/cache_pool_worker_dir_reachqa_20251222"
    os.makedirs(cache_result_base_dir, exist_ok=True)

    """
    启动开关：处理 GRPO RL Parquet 训练集数据（2/2）ReachQA Train 数据集
    """
    res_raw_data_dict_list = read_jsonl_dict_list(input_raw_jsonl_filepath, cache_dir=cache_result_base_dir)
    print(f"Gathered {len(res_raw_data_dict_list)} testcase(s) for ReachQA training dataset")
    if len(res_raw_data_dict_list) > 0:
        print("Cache Dir:", cache_result_base_dir)
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        print("multiprocessing.set_start_method('spawn', force=True)")
        res_ir_dict_list = submit_to_executor_pool(working_func=worker_parse_chart_spec, 
                                                   args_list=res_raw_data_dict_list,
                                                   max_workers=290,
                                                   process_name="Parse ReachQA Train")
    print("DONE")


#  Step 0/2：将 verl/utils/reward_score 目录下的 math.py 重命名为 math_utils.py （防止 verl 包与 math 包冲突）
#
#  Step 0/2: 删除之前残留进程
#     $ kill $(ps aux | grep "chart2code_mllm.example_extract_final_ir" | awk '{print $2}' | tr '\n' ' ') 
#     $ kill $(ps aux | grep "svgr1/bin/python" | awk '{print $2}' | tr '\n' ' ')
#     $ kill $(ps aux | grep "chartmimic/bin/python" | awk '{print $2}' | tr '\n' ' ') 
#
#  Step 0/2：（可选）删除旧缓存，特别是LLM Prompt变更后
#     $ rm -r 旧缓存目录
#
#
# Step 1/2：进入 package 根路径（这里包冲突最小，可以直接调用现有代码）
#     $ cd ~/dmc61-workspace/veRL-MLLM-GRPO/verl/utils/reward_score
# 
# Step 2/2：运行（注意pip升级tqdm到最新版，否则数据太多不显示进度条而卡死）
#   防断电nohup后台运行
#     $ rm nohup.out
#     $ conda activate svgr1
#     $ pyclean . && nohup python -m chart2code_mllm.example_extract_final_ir &
#
#  查看执行结果：进入进程池缓存目录： ${cache_result_base_dir}，找到对应的 case id 的 json 文件
#
#
# P.S., 当所有的数据 LLM IR + Code IR 提取完成后，在GRPO 训练之前，记得把 verl 框架的 math 依赖包恢复回到原来名字
#       即目录 veRL-MLLM-GRPO/verl/utils/reward_score/ 中的 math_utils.py 改回 math.py 
#
# 展示最新10条修改文件
#  $ ls -alrt --time-style=+%Y%m%d%H%M%S | tail -n 10
