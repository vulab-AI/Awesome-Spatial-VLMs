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
from typing import Callable, Dict, List, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, gcot_compute_score


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class GcotRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str):
        self.tokenizer = tokenizer
        if compute_score == 'gcot':
            self.compute_score: Callable[[str, str, str], RewardScore] = gcot_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, data_item in enumerate(data):
            prompt_ids = data_item.batch['prompts']
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            attention_mask = data_item.batch["attention_mask"]

            # prompt_length = prompt_ids.shape[-1]
            # valid_prompt_length = attention_mask[:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_prompt_ids = data_item.non_tensor_batch['raw_prompt_ids']       # no left padding and image_pad
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids)                    # no pad
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            question_type = data_item.non_tensor_batch["question_type"]
            data_type = data_item.non_tensor_batch["data_type"]

            # print("Prompt: ", prompt_str)
            # print("Response: ", response_str)
            score = self.compute_score(prompt_str, ground_truth, question_type, data_type)
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
