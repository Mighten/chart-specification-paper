

from typing import Optional
import re

xml_answer_tag_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


# 只要不是数字且不是布尔值，统一按照字符串编辑距离计算
def get_edit_distance(predict_str: str, gt_str: str) -> int:
    n = len(predict_str)
    m = len(gt_str)
    if n * m == 0:
        return n + m
    D = [ [0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if predict_str[i - 1] != gt_str[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)
    return D[n][m]


# 来自 ChartQA Benchmark
def relaxed_acc(prediction: str, target: str,
                    max_relative_change: float = 0.05) -> bool:

    def _to_float(text: str) -> Optional[float]:
        try:
            match = re.search(r'[\d.]+', text.replace(',', ''))
            if match: return float(match.group())
            return None
        except ValueError:
            return None
        
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    
    if prediction_float is not None and target_float is not None:
        if target_float == 0:
            relative_change = abs(prediction_float - target_float)
        else:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        lp = prediction.strip().lower().replace("true", "yes").replace("false", "no")
        tp = target.lower()
        if ("yes" in lp and "yes" in tp) or ("no" in lp and "no" in tp):
            return True
        if lp in tp:
            return True
        return get_edit_distance(predict_str=lp, gt_str=tp) <= 0.05 * len(tp)


def unpack_xml_answer_tag(raw_response_str: str) -> str:
    answers = xml_answer_tag_pattern.findall(raw_response_str)
    if len(answers) > 0:
        return answers[0].strip()
    return ''


def validate_answer_by_chartqa(ground_truth_str: str, raw_response_str: str) -> dict:
    answer_text = unpack_xml_answer_tag(raw_response_str)
    ap_005 = 3 if relaxed_acc(answer_text, ground_truth_str, 0.05) else 0
    ap_01 = 2 if relaxed_acc(answer_text, ground_truth_str, 0.1) else 0
    ap_02 = 1 if relaxed_acc(answer_text, ground_truth_str, 0.2) else 0
    accuracy_check_dict = {
        # 得分范围 [0, 6]
        "score": ap_005 + ap_01 + ap_02,
        "details": {
            "AP@0.05": ap_005,
            "AP@0.1": ap_01,
            "AP@0.2": ap_02,
        }
    }
    return accuracy_check_dict



if __name__ == '__main__':
    def inner_test(test_predict_str, test______gt_str):
        got_edit_distance = get_edit_distance(test_predict_str, test______gt_str)
        print('PT:', test_predict_str)
        print('GT:', test______gt_str)
        print('got edit distance:', got_edit_distance)
        print('tolerance distance:', 0.05 * len(test______gt_str) )
        print('is edit distance within tolerance? -->', got_edit_distance  <= 0.05 * len(test______gt_str) )
        print('')
    
    test______gt_str1 = 'The quick brown fox jumps over a lazy dog.'
    test_predict_str1 = 'The quick brown dog jumps over a lazy fox.'
    inner_test(test_predict_str1, test______gt_str1)

    test______gt_str2 = 'The quick brown fox jumps over a lazy dog.'
    test_predict_str2 = 'The quick brown fox jumps over a lazy dog'
    inner_test(test_predict_str2, test______gt_str2)
