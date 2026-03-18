import logging

from .lowlevel.code_ir_eval import get_code_ir_dict_from_subprocess_stdout

logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])


def syntax_check(python_code_str: str) -> bool:
    """
    语法检查复合 Code IR 提取：
    - 当语法检查通过，返回 {"score": 0.5, "code_ir": {/* ... */}}
    - 当语法检查失败，返回 {"score": -1}
    """
    syntax_check_dict = {
        "score": -1,
    }
    try:
        assert len(python_code_str) > 0
        code_ir_dict = get_code_ir_dict_from_subprocess_stdout(python_code_str)
        assert len(code_ir_dict) > 0
        syntax_check_dict["code_ir"] = code_ir_dict
        syntax_check_dict["score"] = 0.5
    except Exception as e:
        pass
    return syntax_check_dict


def compile_python3(python_code_str: str) -> dict:
    syntax_check_dict = syntax_check(python_code_str)
    return syntax_check_dict
