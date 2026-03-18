import re
from typing import Dict, Tuple, Optional
import logging


logger = logging.getLogger(__file__)


# TODO: Prompt 前面加 <think>，response 开头拼接上 <think>
def validate_response_structure(raw_response_str: str) -> dict:
    validation_error_message_list = []
    format_score = 0

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = raw_response_str.count(tag_str)
        # find the rightmost index
        positions[tag_name] = raw_response_str.rfind(tag_str)
        if count != expected_count:
            format_score -= 1
            validation_error_message_list.append(f"`{tag_str}` appears {count} time(s) (expected {expected_count})")
        else:
            format_score += 1

    # Verify tag order
    if positions['think_start'] < positions['think_end']:
        format_score += 1
    else:
        validation_error_message_list.append(f"<think> NOT before </think>")
        format_score -= 1
    
    if positions['answer_start'] < positions['answer_end']:
        format_score += 1
    else:
        validation_error_message_list.append(f"<answer> NOT before </answer>")
        format_score -= 1

    if not (positions['think_start'] < positions['think_end'] < positions['answer_start'] < positions['answer_end']):
        format_score -= 1
        validation_error_message_list.append(f"Check Failed: <think>(pos={positions['think_start']}), </think>(pos={positions['think_end']}), <answer>(pos={positions['answer_start']}), </answer>(pos={positions['answer_end']})")
    else:
        format_score += 1

    format_score_dict = {
        # 将得分缩放到 [0, 2]
        'score': round(format_score / 7, 2) + 1,
        'error_msg': validation_error_message_list
    }
    return format_score_dict
