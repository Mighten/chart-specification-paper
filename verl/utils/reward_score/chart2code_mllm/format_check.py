
import logging


logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])


def validate_response_structure(raw_response_str: str, enable_thinking: bool = True) -> dict:
    # 关闭 CoT 思考检查特殊处理：
    if not enable_thinking:
        python_code_begin_idx, python_code_end_idx = raw_response_str.rfind('```python'), raw_response_str.rfind('```')
        # 正确格式提前返回0：```python\n{code}\n``` 
        if python_code_begin_idx != -1 and python_code_end_idx != -1 and python_code_begin_idx < python_code_end_idx:
            return {'score': 0}
        # 否则，异常格式返回-2
        return {'score': -2}

    validation_error_message_list = []

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
            validation_error_message_list.append(f"`{tag_str}` appears {count} time(s) (expected {expected_count})")

    # Verify tag order
    if positions['think_start'] > positions['think_end']:
        validation_error_message_list.append(f"<think> NOT before </think>")
    
    if positions['answer_start'] > positions['answer_end']:
        validation_error_message_list.append(f"<answer> NOT before </answer>")

    python_code_begin_idx, python_code_end_idx = raw_response_str.rfind('```python'), raw_response_str.rfind('```')
    if not (positions['think_start'] < positions['think_end'] < positions['answer_start'] < python_code_begin_idx < python_code_end_idx < positions['answer_end']):
        validation_error_message_list.append(f"Check Failed: <think>(pos={positions['think_start']}), </think>(pos={positions['think_end']}), <answer>(pos={positions['answer_start']}), ```python(pos={python_code_begin_idx}), ```(pos={python_code_end_idx}), </answer>(pos={positions['answer_end']})")

    format_score = -2 if len(validation_error_message_list) > 0 else 0
    format_score_dict = {
        'score': format_score,
        # 'error_msg': validation_error_message_list
    }
    return format_score_dict
