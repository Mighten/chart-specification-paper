import base64
import http
import json
import requests
import random
import os
import time
import json
import logging
import gc

logger = logging.getLogger(__file__)


provider_url_pool = [
    # Huawei vllm-ascend 部署 95
    "http://10.170.23.95:8000/v1/chat/completions",

    # Huawei vllm-ascend 部署 87
    "http://10.170.23.87:8000/v1/chat/completions",
]

# 多进程调用时，使用系统随机源重新初始化随机数种子
# random.seed(os.urandom(8))

# 配置常量（旧接口配置）
LLM_MODEL = "Qwen3-32B"
DEFAULT_HEADERS = {"Content-Type": "application/json"}


logger = logging.getLogger(__file__)

# 将制定文件编码为 BASE64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_llm(query: str, timeout_s=420.0, failure_backoff_base_s=3, max_retries=10) -> str:
    payload_obj = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "temperature": 0.0,
        "n": 1,
        "stream": False,
    }

    payload = json.dumps(payload_obj, ensure_ascii=False)

    retries = 0
    provider_url = ''
    while True:
        try:
            # 每次尝试前，从地址池中随机选择 provider_url
            random.seed(os.urandom(8))
            provider_url = random.choice(provider_url_pool)
            with requests.post(url=provider_url, headers=DEFAULT_HEADERS, data=payload, timeout=timeout_s, stream=False) as response:
                # 显式设置编码方式 UTF-8 ，防止中文显示乱码
                response.encoding = 'utf-8'
                if response.status_code != http.HTTPStatus.OK:
                    raise Exception(
                        f'Failed to request {provider_url} with status {response.status_code}: `{repr(response.text)}`')
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(
                    f'max retries exceeded for Qwen3-32B url `{provider_url}`, return empty string `` instead.',
                    exc_info=True)
                return ''  # 返回空白字符串
            # 随机时间退避
            logger.warning(f'failed due to {str(e)}, retry {retries}-th for the Qwen3-32B query {repr(query)}')
            time.sleep(random.uniform(0.0, failure_backoff_base_s))


def get_qwen_v2_5_72b_answer(query):
    return call_llm(query)


if __name__ == '__main__':
    print(get_qwen_v2_5_72b_answer('Hello'))
