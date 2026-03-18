

import logging
import json

from . import accuracy_check, format_check

logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])


def compute_score_with_cot(raw_answer_str: str, ground_truth_str: str) -> float:
    total_score = 0.0
    # 回答格式校验分 [0, 2]，格式不对，提前拦截
    #   备注：格式不对认为是 < 1.9 分（防止浮点数误差） 
    format_check_result_dict = format_check.validate_response_structure(raw_answer_str)
    # TODO: < 0: break, return -1
    if format_check_result_dict['score'] < 1.9:
        total_score_dict = {
            # total_score 范围 [0, 8]，缩放到 [-1, 1]
            'score': -1,
            'details': {
                "format": format_check_result_dict,
                "raw_answer": raw_answer_str,
                "ground_truth": ground_truth_str,
            }
        }
        logger.error(f'[ChartQA](Malformed) Final Reward: {json.dumps(total_score_dict, ensure_ascii=False)}')
        return total_score

    # 消融实验1：不奖励 thinking 内容质量

    # 格式奖励（-1/ 0）与 answer [0, 1] 区分开
    total_score += format_check_result_dict['score']
    # 回答准确率校验 [0, 6] 分
    accuracy_check_dict = accuracy_check.validate_answer_by_chartqa(ground_truth_str, raw_answer_str)
    total_score += accuracy_check_dict['score']
    # 日志显示过程
    total_score_dict = {
        # total_score 范围 [0, 8]，缩放到 [-1, 1]
        'score': round(total_score / 4 - 1, 3),
        'details': {
            "format": format_check_result_dict,
            "accuracy": accuracy_check_dict,
            "raw_answer": raw_answer_str,
            "ground_truth": ground_truth_str,
        }
    }
    # logger.warning(f'[ChartQA] Final Reward: {json.dumps(total_score_dict, ensure_ascii=False)}')
    return total_score


def compute_score(raw_answer_str: str, ground_truth_str: str) -> float:
    return compute_score_with_cot(raw_answer_str, ground_truth_str)
