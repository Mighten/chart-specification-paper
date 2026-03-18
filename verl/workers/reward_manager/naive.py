# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import json
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

import logging
import gc

logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.process_pool_worker_num = 310   # 定义进程池工作进程个数 310 - 每一路 rollout 都是一个单独的进程
        logger.warn(f"[Mingchen] ProcessPool initialized with worker num {self.process_pool_worker_num}")


    def process_pool_worker_get_reward_score(self, param_dict: dict):
        """
            Process Pool worker function: submit to ProcessPoolExecutor for batch processing
        """
        valid_response_ids = param_dict['valid_response_ids']
        data_source = param_dict['data_source']
        extra_info = param_dict['extra_info']
        ground_truth = param_dict['ground_truth']

        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        reward_score_obj = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        valid_response_length = len(valid_response_ids)
        return (reward_score_obj, valid_response_length)


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 准备 ProcessPoolExecutor worker func param，禁止多进程访问 CUDA tensor
        # Accessing CUDA tensor object is ONLY allowed in main process & main thread
        # !!! DO NOT access any CUDA object concurrently !!! 
        #    otherwise the `RuntimeError: CUDA error: an illegal memory access was encountered` is to be TRIGGERED !!!
        multiprocessing_worker_func_param_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            multiprocessing_worker_func_param_list.append({
                'data_source': data_item.non_tensor_batch[self.reward_fn_key],
                'extra_info': data_item.non_tensor_batch.get("extra_info", None),
                'ground_truth': data_item.non_tensor_batch["reward_model"]["ground_truth"],
                'valid_response_ids': valid_response_ids,
            })
            
        # future 对象映射 index
        future_obj_to_index_dict = {}
        # future 对象 index 映射到结果tuple (reward_score, valid_response_length)
        future_obj_id_to_reward_dict = {}
        with ProcessPoolExecutor(max_workers=self.process_pool_worker_num) as executor:
            for i in range(len(data)):
                param_dict = multiprocessing_worker_func_param_list[i]
                future_obj = executor.submit(self.process_pool_worker_get_reward_score, param_dict)
                future_obj_to_index_dict[future_obj] = i
            
            for future_obj in as_completed(future_obj_to_index_dict.keys()):
                i = future_obj_to_index_dict[future_obj]
                reward_score_obj, valid_response_length = future_obj.result()
                if isinstance(reward_score_obj, dict):
                    reward = reward_score_obj["score"]
                    # Store the information including original reward
                    for key, value in reward_score_obj.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = reward_score_obj
                
                # 通过字典转存reward score，禁止在进程池中向 reward_tensor 赋值，
                # 否则引起 RuntimeError: CUDA error: an illegal memory access was encountered
                future_obj_id_to_reward_dict[i] = (reward, valid_response_length)
        
        future_obj_to_index_dict.clear()
        multiprocessing_worker_func_param_list.clear()
        
        # Assigning reward scores to CUDA object `reward_tensor` is only allowed in single process & single thread.
        for i, reward_score_res in future_obj_id_to_reward_dict.items():
            reward_score, valid_response_length =  reward_score_res
            reward_tensor[i, valid_response_length - 1] = reward_score
        
        future_obj_id_to_reward_dict.clear()
        gc.collect()  # 强制回收主进程引用
        torch.cuda.empty_cache()

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
