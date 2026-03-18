

import logging
import json
import re
import numpy as np

from . import evaluate_spec, format_check, compile_check

logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])



def unpack_xml_answer_tag(raw_response_str: str) -> str:
    xml_answer_tag_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answers = xml_answer_tag_pattern.findall(raw_response_str)
    if len(answers) > 0:
        return answers[0].strip()
    return ''


def unpack_python_code_str(raw_answer_str: str) -> str:
    matcher = re.search(r"```python\n(.*?)\n```", raw_answer_str, re.DOTALL)
    return matcher.group(1).strip() if matcher else ''


def unpack_yaml_codefence(raw_answer_str: str) -> str:
    matcher = re.findall(r"```yaml\n(.*?)\n```", raw_answer_str, flags=re.DOTALL)
    return matcher.group(1).strip() if matcher else ''


def compute_score(raw_answer_str: str, ground_truth_str: str, extra_info_dict: dict, inject_code_ir_dict: dict = None) -> float:
    """
        chart2code
        - 格式检查：失败直接返回-2，通过则忽略
        - 编译检查：失败直接返回-1，通过则忽略
        - 提取 spec 评估准确率 [0, 1]
    """
    
    dataset_split_name = extra_info_dict.get('split', 'train')
    dataset_item_id = extra_info_dict.get('id', 0)
    # 若缺省，则认为开启Thinking 模式：先 <think></think> 再 <answer>```pyhton\n{code}\n```\n</answer>
    #         若显示指定 enable_thinking=False，则直接输出 ```python\n{code}\n```
    enable_thinking = extra_info_dict.get('enable_thinking', True)
    spec_dict = json.loads(ground_truth_str)

    total_score = 0

    # 1. 回答格式校验
    format_check_result_dict = format_check.validate_response_structure(raw_answer_str, enable_thinking=enable_thinking)
    if np.isnan(format_check_result_dict['score']):
        logger.error(f"compute_score of format check returns NaN in chart2code #{dataset_item_id}")
        format_check_result_dict['score'] = -2
    total_score += format_check_result_dict['score']

    # 回答格式格式校验失败，提前结束计算reward
    if format_check_result_dict['score'] < 0:
        total_score_dict = {
            'score': total_score,
            'details': {
                "format": format_check_result_dict["score"],
                # "raw_answer": raw_answer_str,
            },
        }
        # 只打印格式错误日志信息，用于衡量偏离基准模型的程度
        logger.error(f'[chart2code]{"" if enable_thinking else "[No Think Mode]"}[{dataset_split_name}-{dataset_item_id}](Malformed) Final Reward: {total_score}')
        return total_score

    # 2. 编译检查 [-1, 0]
    if enable_thinking:
        # 若开启了 CoT 生成：先<think></think> 再 <answer>```python\n{code}\n```</answer> 保证只提取 <answer></answer> 内部的 Python 代码块
        raw_answer_str = unpack_xml_answer_tag(raw_answer_str)
    python_code_str = unpack_python_code_str(raw_answer_str)
    compile_check_result_dict = compile_check.compile_python3(python_code_str)
    if np.isnan(compile_check_result_dict['score']):
        logger.error(f"[NaN] compute_score of compile check returns NaN in chart2code #{dataset_item_id}")
        compile_check_result_dict['score'] = -1
    total_score += compile_check_result_dict['score']
    if compile_check_result_dict['score'] < 0:
        total_score_dict = {
            'score': total_score,
            'details': {
                "format": format_check_result_dict["score"],
                "compile": compile_check_result_dict["score"],
                # "raw_answer": raw_answer_str,
            },
        }
        logger.error(f'[chart2code][{dataset_split_name}-{dataset_item_id}](Compile Failed) Final Reward: {total_score}')
        return total_score

    # 从编译检查步骤直接抽取 Code IR
    code_ir_spec = compile_check_result_dict.get("code_ir", {})

    # 3. 回答准确率校验：[0, 10]
    accuracy_check_dict = evaluate_spec.evaluate_chart2code_spec_by_ir_new(python_code_str, spec_dict, code_ir_dict=code_ir_spec)
    # logger.warning(f"{accuracy_check_dict=!r}")
    if np.isnan(accuracy_check_dict['score']):
        logger.error(f"compute_score of accuracy check returns NaN in chart2code #{dataset_item_id}")
        accuracy_check_dict['score'] = 0
    total_score += accuracy_check_dict['score']

    # 日志显示过程
    total_score_dict = {
        'score': total_score,
        'details': {
            "format": format_check_result_dict['score'],
            "compile": compile_check_result_dict['score'],
            "accuracy": accuracy_check_dict['score'],
            # "raw_answer": raw_answer_str,
            # "gt_spec": spec_dict,
        },
    }
    # logger.warning(f'[chart2code][{dataset_split_name}-{dataset_item_id}] Final Reward: {json.dumps(total_score_dict, ensure_ascii=False)}')
    return total_score
