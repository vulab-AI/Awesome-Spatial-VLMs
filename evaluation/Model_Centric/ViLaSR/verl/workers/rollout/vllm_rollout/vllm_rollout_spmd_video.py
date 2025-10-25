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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import os
from contextlib import contextmanager
from typing import Any, List, Union, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, AutoProcessor
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

import copy
import pdb
from PIL import Image
from qwen_vl_utils import fetch_image
from ....models.transformers.qwen2_vl import get_rope_index
# from ....models.transformers.qwen2_5_vl import get_rope_index


from  ....utils.edit_image_video import merge_bbox_movement, parse_bbox_and_movement, plot_movement, plot_bounding_boxes
import json

MAX_IMAGES=42

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


from dataclasses import dataclass
@dataclass
class ProcessData:
    index: int
    response: str
    response_ids: List
    mm_data: Dict
    bbox_list_origin: List
    movement_list_origin: List
    finish_reason: str
    is_finished: bool
    grid_size: int

def calculate_grid_centers(image_size=616, grid_size=5):
    # matplotlib的默认边距
    margin_left = int(image_size * 0.125)    # 77 pixels
    margin_right = int(image_size * 0.1)     # 62 pixels
    margin_bottom = int(image_size * 0.11)   # 68 pixels
    margin_top = int(image_size * 0.12)      # 74 pixels
    
    # 计算实际可用空间
    usable_width = image_size - (margin_left + margin_right)
    usable_height = image_size - (margin_top + margin_bottom)
    
    # 计算每个格子的大小
    cell_width = usable_width / grid_size
    cell_height = usable_height / grid_size
    
    # 计算所有格子中心的坐标
    centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算中心坐标，考虑不同的边距
            center_x = margin_left + cell_width/2 + j * cell_width
            center_y = margin_top + cell_height/2 + i * cell_height
            centers.append((center_x, center_y))
    # print("calculate_grid_centers:", image_size, margin_left, margin_right, cell_width, cell_height)
    return centers, (cell_width+cell_height)/2  # 返回x和y方向的格子大小

def check_path_tracer(movement_list, centers, cell_size):
    for movement in movement_list:
        for key in ['start_point_2d', 'end_point_2d']:
            x, y = int(movement[key][0]), int(movement[key][1])
            min_distance = min([np.sqrt((x-c[0])**2 + (y-c[1])**2) for c in centers])
            if min_distance > cell_size/2:
                # print(key, movement[key], min_distance, cell_size/2, centers)
                return False
    return True


def check_repetition(allindex, bbox_list_origin, movement_list_origin):
    # print(allindex)
    # print(bbox_list_origin)
    # print(movement_list_origin)
    for cnt, tmp_index in enumerate(allindex):
        for bbox_list in list(bbox_list_origin.values()):
            for bbox in bbox_list:
                if bbox in allindex[tmp_index]["bbox_list"]:
                    return True
        for movement_list in list(movement_list_origin.values()):
            for movement in movement_list:
                if movement in allindex[tmp_index]["movement_list"]:
                    return True
    return False


def process_single_response(data: ProcessData):
    """处理单个响应的函数"""
    if data.is_finished is True:
        return {
            'index': data.index,
            'response': data.response,
            'response_ids': data.response_ids,
            'finish_reason': data.finish_reason,
            'is_finished': data.is_finished,
            'processed_image_idx': [None],
        }
    try:
        # 解析和绘图
        bbox_list_new, movement_list_new = parse_bbox_and_movement(data.response)
        current_image_index = len(data.mm_data['image'])
        image_index_list, image_list = [], []
        bbox_list, movement_list = data.bbox_list_origin, data.movement_list_origin
        finish_reason = None
        allindex = {}
        try:
            for tmp_bbox_list in bbox_list_new:
                tmp_bbox_list = copy.deepcopy(tmp_bbox_list)
                if tmp_bbox_list["index"] in allindex:
                    if "bbox_list" in allindex[tmp_bbox_list["index"]]:
                        allindex[tmp_bbox_list["index"]]["bbox_list"].append(tmp_bbox_list)
                    else:
                        allindex[tmp_bbox_list["index"]]["bbox_list"] = [tmp_bbox_list]
                else:
                    allindex[tmp_bbox_list["index"]] = {'bbox_list': [tmp_bbox_list], 'movement_list': []}
            for tmp_movement_list in movement_list_new:
                tmp_movement_list = copy.deepcopy(tmp_movement_list)
                if tmp_movement_list["index"] in allindex:
                    if "movement_list" in allindex[tmp_movement_list["index"]]:
                        allindex[tmp_movement_list["index"]]["movement_list"].append(tmp_movement_list)
                    else:
                        allindex[tmp_movement_list["index"]]["movement_list"] = [tmp_movement_list]
                else:
                    allindex[tmp_movement_list["index"]] = {'bbox_list': [], 'movement_list': [tmp_movement_list]}
        except Exception as e:
            traceback.print_exc()
            print("bbox_list_new, movement_list_new: ", bbox_list_new, movement_list_new)
            finish_reason = "ToolGenError"
        # if len(allindex) > 1:
        #     print(data.index)
        # # allindex[16] = {'bbox_list': [{'index': 16, 'bbox_2d': [10, 10, 500, 500], 'label': 'staircase with railings'}], 'movement_list': []}
        # print(len(allindex), allindex)

        if len(allindex) == 0:
            finish_reason = "ToolError"
        # elif len(allindex) > 3:
        #     finish_reason = "TooManyTools"
        elif len(data.mm_data['image']) >= MAX_IMAGES+1:
            finish_reason = "TooManyImages"
        # elif check_repetition(allindex, bbox_list, movement_list):
        #     finish_reason = "Repetition"

        if finish_reason is not None:
            return {
                'index': data.index,
                'processed_image_idx': [None],
                'image': [data.mm_data['image'][0].copy()],
                'response': data.response,
                'response_ids': data.response_ids,
                'finish_reason': finish_reason,
                'bbox_list': bbox_list,
                'movement_list': movement_list,
                'is_finished': True,
            }
        for cnt, tmp_index in enumerate(allindex):
            bbox_list_new, movement_list_new = allindex[tmp_index]["bbox_list"], allindex[tmp_index]["movement_list"]
            image_index_new = current_image_index + cnt
            # print("bbox_list_new: ", bbox_list_new)
            # print("movement_list_new: ", movement_list_new)
            image_index, bbox_list, movement_list = merge_bbox_movement(
                bbox_list_origin=data.bbox_list_origin,
                movement_list_origin=data.movement_list_origin,
                bbox_list_new=bbox_list_new,
                movement_list_new=movement_list_new,
                image_index_new=image_index_new,
            )
            image_index_list.append(image_index)
            if image_index == -1:
                return {
                    'index': data.index,
                    'processed_image_idx': [None],
                    'image': [data.mm_data['image'][0].copy()],
                    'response': data.response,
                    'response_ids': data.response_ids,
                    'finish_reason': "ToolError",
                    'bbox_list': bbox_list,
                    'movement_list': movement_list,
                    'is_finished': True,
                }
            # print("data.bbox_list_origin:", data.bbox_list_origin)
            # print("data.movement_list_origin:", data.movement_list_origin)
            # print("image_index:", image_index)
            # print("bbox_list: ", bbox_list)
            # print("movement_list: ", movement_list)

            # print("bbox_list: ", bbox_list)
            # print("movement_list: ", movement_list)

            # 复制图像以避免多线程冲突
            try:
                image = data.mm_data['image'][image_index].copy()
                assert isinstance(image, Image.Image)
                input_width, input_height = image.size

                # 绘制边界框和移动路径
                plot_bounding_boxes(image, bbox_list[image_index_new], input_height=input_height, input_width=input_width)
                plot_movement(image, movement_list[image_index_new], input_height=input_height, input_width=input_width)
                image_list.append(image)

            except IndexError as e:
                print(f"Image index error: {str(e)}")
                return {
                    'index': data.index,
                    'processed_image_idx': [None],
                    'image': [data.mm_data['image'][0].copy()],
                    'response': data.response,
                    'response_ids': data.response_ids,
                    'finish_reason': "IndexError",
                    'bbox_list': bbox_list,
                    'movement_list': movement_list,
                    'is_finished': True,
                }
        
        return {
            'index': data.index,
            'processed_image_idx': image_index_list,
            'image': image_list,
            'response': data.response,
            'response_ids': data.response_ids,
            'finish_reason': data.finish_reason,
            'bbox_list': bbox_list,
            'movement_list': movement_list,
            'is_finished': data.is_finished
        }
    except Exception as e:
        print(f"Error processing response {data.index}: {str(e)}")
        traceback.print_exc()
        return None

def save_samples_info(samples_info, save_dir):

    def get_unique_dir(base_path, prefix='generation'):
        """generate unique dirctory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 0
        while True:
            if counter == 0:
                dir_name = f"{prefix}_{timestamp}"
            else:
                dir_name = f"{prefix}_{timestamp}_{counter}"

            full_path = os.path.join(base_path, dir_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    all_sample_dir = []
    for idx, sample in enumerate(samples_info):
        # 为每个样本创建子目录
        sample_dir = get_unique_dir(save_dir, f'sample')
        os.makedirs(sample_dir, exist_ok=True)
        all_sample_dir.append(sample_dir)

        # 保存文本信息
        text_data = {
            'prompt': sample['prompt'],
            'sequence': sample['sequence'],
            'response': sample['response'],
            'finish_reason': sample['finish_reason'],
            'execution_pass': sample['execution_pass']
        }
        
        with open(os.path.join(sample_dir, 'text_data.json'), 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        # 保存图片
        if 'multi_modal_data' in sample and 'image' in sample['multi_modal_data']:
            images_dir = os.path.join(sample_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            for img_idx, img in enumerate(sample['multi_modal_data']['image']):
                if isinstance(img, Image.Image):
                    img_path = os.path.join(images_dir, f'image_{img_idx}.png')
                    img.save(img_path)
    return all_sample_dir


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer, processor: AutoProcessor):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor                  # add processor
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=int((config.prompt_length + config.response_length)*1.5),                # 模型支持的最大上下文长度（单个序列的最大长度）
            max_num_batched_tokens=config.max_num_batched_tokens,                       # 批量推理中所有序列的 token 总数上限
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        # sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        sampling_kwargs = {"max_tokens": config.single_turn_response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _get_multi_turn_mask(self, response_tokens):
        """
        生成多轮对话的attention mask，mask掉所有特殊标记和提示部分
        
        Args:
            response_tokens: 包含多轮对话的token序列
            
        Returns:
            attention_mask: 与response_tokens同样大小的mask，只保留助手的响应内容
        """
        
        # 获取特殊token的id
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = self.tokenizer.convert_tokens_to_ids("user")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        pad_id = self.tokenizer.pad_token_id
        newline_id = 198

        attention_mask = torch.zeros_like(response_tokens)  # 初始化全为0
        current_pos = 0
        in_assistant_response = True  # 初始状态为True，因为从assistant响应开始
        while current_pos < len(response_tokens):
            if response_tokens[current_pos] == im_end_id:
                # 遇到im_end_id，切换状态
                in_assistant_response = False
                current_pos += 1
                continue
                
            if (current_pos + 2 < len(response_tokens) and 
                response_tokens[current_pos] == im_start_id and 
                response_tokens[current_pos + 1] == assistant_id and
                response_tokens[current_pos + 2] == newline_id):
                # 找到新的assistant响应开始（包括换行符）
                in_assistant_response = True
                current_pos += 3  # 跳过im_start, assistant和换行符
                continue
                
            if in_assistant_response and response_tokens[current_pos] != pad_id:
                # 在assistant响应内容中，且不是padding
                attention_mask[current_pos] = 1
                
            current_pos += 1

        return attention_mask

    def decode_masked_tokens(self, input_ids, mask, prompt_len):
        """
        对mask非0部分进行detokenize
        
        Args:
            input_ids: 输入token序列
            mask: attention mask
            prompt_len: prompt长度
        """
        # 获取response部分的tokens和mask
        response_tokens = input_ids[prompt_len:self.config.response_length]
        response_mask = mask.bool()  # 转换为布尔掩码
        
        # 收集所有需要decode的token段 
        valid_segments = []
        current_segment = []
        
        for token, is_valid in zip(response_tokens, response_mask):
            if is_valid:
                current_segment.append(token.item())
            elif current_segment:  # 当前段结束
                valid_segments.append(current_segment)
                current_segment = []
        
        if current_segment:  # 处理最后一个段
            valid_segments.append(current_segment)
        
        # 对每个有效段进行decode
        decoded_segments = []
        for segment in valid_segments:
            decoded_text = self.tokenizer.decode(segment, skip_special_tokens=True)
            decoded_segments.append(decoded_text)
        
        return decoded_segments
    
    def decode_masked_tokens_2(self, input_ids, mask, prompt_len):
        """
        将mask==1部分替换为pad_token，其余部分保留
        
        Args:
            input_ids: 输入token序列
            mask: attention mask
            prompt_len: prompt长度
        """
        # 获取response部分的tokens和mask
        response_tokens = input_ids[prompt_len:self.config.response_length].clone()  # 创建副本
        response_mask = mask.bool()
        
        # 获取pad token id
        pad_token_id = self.tokenizer.pad_token_id
        
        # 将mask==1的部分替换为pad_token
        response_tokens[response_mask] = pad_token_id
        
        # decode整个序列
        decoded_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
        
        # 同时decode原始序列用于对比
        original_text = self.tokenizer.decode(
            input_ids[prompt_len:self.config.response_length], 
            skip_special_tokens=False
        )
        
        return {
            'masked_text': decoded_text,
            'original_text': original_text
        }


    def _get_prompts_and_indices(self, samples_info, num_llm_calls_available_list):
        prompts, multi_modal_data, indices=[], [], []
        for index, (info, num_calls) in enumerate(zip(samples_info, num_llm_calls_available_list)):
            if not info['stop'] and len(info['multi_modal_data']['image']) <= MAX_IMAGES and num_calls>=0:
                prompts.append(info['sequence'])
                multi_modal_data.append(info['multi_modal_data'])
                indices.append(info['index'])
        return prompts, multi_modal_data, indices


    def check_token_length(self, sequence, images):
        """检查当前序列的总token长度"""
        # 计算文本token
        text_tokens = len(self.tokenizer.encode(sequence, add_special_tokens=False))
        # 估算图片token (根据实际模型调整)
        image_tokens = len(images) * 576  # 假设每张图片大约占用576个token
        return text_tokens + image_tokens

    def _multi_turn_generate(self, vllm_inputs=None, sampling_params=None, prompt_token_ids=None, use_tqdm=False, save_dir=None):
        sampling_params=copy.deepcopy(sampling_params)
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            prompt = self.tokenizer.decode(single_vllm_input['prompt_token_ids'], skip_special_tokens=False)            # vllm use raw_prompt_ids
            new_vllm_inputs.extend([{
                "prompt": prompt,
                "multi_modal_data": copy.deepcopy(single_vllm_input['multi_modal_data']),           # must use deepcopy
                "grid_size": single_vllm_input['grid_size'],
            }   for _ in range(sampling_params.n)])
        sampling_params.n=1
        sampling_params.detokenize=True             # True convert ids to text
        samples_info = []
        for index, item in enumerate(new_vllm_inputs):
            origin_image = item['multi_modal_data']['image'][0]
            width, height = origin_image.size
            assert width % 28==0 and height % 28==0, f"Image dimensions ({width}, {height}) must be divisible by 28. " \
                f"Current remainder: width={width%28}, height={height%28}"
            # processed_image = fetch_image({'image': origin_image})           # qwen fetch_image for 14倍数
            processed_image:dict = item['multi_modal_data']                                # keep the same resolutions
            # processed_image = [fetch_image({'image': origin_image}) for origin_image in item['multi_modal_data']['image']]
            sample_info = {
                "prompt": item["prompt"],
                "sequence": item["prompt"],
                "multi_modal_data": processed_image,          #  {"image": processed_image}, "multi_modal_data": dict[str, list[Image]]
                "response": "",
                "stop": False,
                "finish_reason": None,
                "processed_image_idx": [],
                "index": index,
                "mask_info": [],
                "execution_pass": 0,
                "bbox_list": {img_idx: [] for img_idx in range(len(processed_image['image']))},          # save bbox_list for each image index 
                "movement_list": {img_idx: [] for img_idx in range(len(processed_image['image']))},      # save movement_list for each image index 
                "grid_size": item['grid_size'],
                "total_response_tokens": 0,                                                             # 添加response token长度跟踪
            }
            samples_info.append(sample_info)
        intermediate_prompt = 'The index of the given image is {current_image_idx} (width: {width}, height: {height}). Continue your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:'
        final_prompt = 'The index of the given image is {current_image_idx} (width: {width}, height: {height}). Then, you can not invoke the Object Mapper or Path Tracer tool. Please answer the initial question and structure your response as required:'
        intermediate_template = """<|im_end|>
<|im_start|>user
{pad}
{prompt}
<|im_end|>
<|im_start|>assistant
"""     
        num_llm_calls_available=copy.deepcopy(self.config.num_llm_calls_available) - 1 
        num_llm_calls_available_list = [
            num_llm_calls_available + 10 if sample['grid_size'] is not None else num_llm_calls_available 
            for sample in samples_info]

        while any(num_calls >= 0 for num_calls in num_llm_calls_available_list):
            print("total_response_tokens: ", [x["total_response_tokens"] for x in samples_info])
            input_prompts, multi_modal_data, indices=self._get_prompts_and_indices(samples_info, num_llm_calls_available_list)  # _get_prompts_and_indices
            # num_llm_calls_available-=1
            for i in range(len(num_llm_calls_available_list)):      # 更新每个样本的调用次数
                if num_llm_calls_available_list[i] >= 0:
                    num_llm_calls_available_list[i] -= 1
            # print("multi_modal_data len: ", [len(multi_modal_data[i]['image']) for i in range(len(multi_modal_data))])
            input_prompts = [{
                    'prompt_token_ids': self.tokenizer.encode(prompt, add_special_tokens=False)[:self.config.prompt_length+self.config.response_length],
                    'multi_modal_data': mm_data
                } for prompt, mm_data in zip(input_prompts, multi_modal_data)]

            print(f'num_llms_calls_available: {num_llm_calls_available_list}', [len(x['prompt_token_ids']) for x in input_prompts])
            print("multi_modal_data size: ", [x['multi_modal_data']['image'][0].size for x in input_prompts], "\nmulti_modal_data len: ", [len(x['multi_modal_data']['image']) for x in input_prompts])
            outputs = self.inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses=[x.outputs[0].text for x in sorted_outputs]
            output_token_ids = [x.outputs[0].token_ids for x in sorted_outputs]  # 获取生成的token ids
            finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]  # "stop", "length"
            stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]      # None: have EOS

            for i, index in enumerate(indices):
                new_tokens = len(output_token_ids[i])
                if samples_info[index]['total_response_tokens'] + new_tokens >= self.config.response_length:
                    finish_reason[i] = 'max_tokens'
                    stop_reason[i] = None 

            if all(num_calls == -1 for num_calls in num_llm_calls_available_list):          # last turn
                for i ,index in enumerate(indices):
                    samples_info[index]['response']+=responses[i]
                    samples_info[index]['sequence']+=responses[i]
                    samples_info[index]['stop']=True
                    samples_info[index]['finish_reason']=finish_reason[i]
                break

            def _is_finished(finish_reason, stop_reason, response):
                if finish_reason=='stop' and stop_reason is None and "<answer>" in response and "</answer>" in response:            # finish one complete response and output answer
                    return True
                if finish_reason=='length':         # single turn trunction
                    return True
                if finish_reason=='rule':
                    return True
                if finish_reason=='max_tokens':     # total response trunction
                    return True
                return False
            is_finished=[_is_finished(finish_reason[i], stop_reason[i], responses[i]) for i in range(len(finish_reason))]
            if all([x for x in is_finished]):                   # check if all samples are finished
                for i ,index in enumerate(indices):
                    samples_info[index]['response']+=responses[i]
                    samples_info[index]['sequence']+=responses[i]
                    samples_info[index]['stop']=True
                    samples_info[index]['finish_reason']=finish_reason[i]
                break
            
            # ----------- Parallel Process -----------
            # Prepare Data
            process_data_list = [
                ProcessData(
                    index=index,
                    response=responses[i],
                    response_ids=output_token_ids[i],
                    mm_data=samples_info[index]['multi_modal_data'],
                    bbox_list_origin=samples_info[index]["bbox_list"],
                    movement_list_origin=samples_info[index]["movement_list"],
                    finish_reason=finish_reason[i],
                    is_finished=is_finished[i],                      # if is_finished == True, stop reasoning
                    grid_size=samples_info[index]['grid_size'],
                )  for i, index in enumerate(indices)] 
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=1 ) as executor:         # max(min(len(indices), os.cpu_count()//2, 64), 1)
            # with ThreadPoolExecutor(max_workers=1) as executor:
                results = list(executor.map(process_single_response, process_data_list))

            # 更新samples_info
            for result in results:
                if result is not None:
                    index = result['index']
                    samples_info[index]['response'] += result['response'] # + intermediate_template.format(prompt=intermediate_prompt)
                    samples_info[index]['stop'] = result['is_finished']
                    samples_info[index]['finish_reason'] = result['finish_reason']
                    samples_info[index]['processed_image_idx'].extend(result['processed_image_idx'])

                    if result['is_finished'] is False:
                        current_image_count = len(samples_info[index]['multi_modal_data']['image']) 
                        if len(result["image"]) > 1:
                            current_image_idx = current_image_count + 1
                            # 计算添加所有图片和prompt需要的token数量
                            current_tokens = samples_info[index]['total_response_tokens']
                            response_tokens = len(result['response_ids'])
                            total_required_tokens = response_tokens
                            pad_prompt = ""
                            images_to_add = []
                            # 预计算所有图片和prompt的token数量
                            for tmp_image_idx, tmp_image in enumerate(result["image"]):
                                width, height = fetch_image({"image": tmp_image}).size
                                # 构建当前图片的prompt
                                if current_image_count + tmp_image_idx + 1 >= MAX_IMAGES:
                                    current_prompt = (f"<|vision_start|><|image_pad|><|vision_end|>" + 
                                                    final_prompt.format(
                                                        current_image_idx=current_image_idx + tmp_image_idx,
                                                        width=width,
                                                        height=height,
                                                    ))
                                    pad_prompt += current_prompt
                                    images_to_add.append(tmp_image)
                                    break
                                else:
                                    if tmp_image_idx <= len(result["image"]) - 2:
                                        current_prompt = f"<|vision_start|><|image_pad|><|vision_end|>The index of the given image is {current_image_idx+tmp_image_idx} (width: {width}, height: {height}).\n"
                                    else:
                                        prompt_template = (intermediate_prompt 
                                                        if num_llm_calls_available_list[index] > 0 
                                                        else final_prompt)
                                        current_prompt = (f"<|vision_start|><|image_pad|><|vision_end|>" + 
                                                        prompt_template.format(
                                                            current_image_idx=current_image_idx + tmp_image_idx,
                                                            width=width,
                                                            height=height,
                                                        ))
                                    pad_prompt += current_prompt
                                    images_to_add.append(tmp_image)
                            
                            # 计算total token
                            template_str = intermediate_template.format(prompt="", pad=pad_prompt)
                            template_tokens = len(self.tokenizer.encode(template_str, add_special_tokens=False))
                            image_tokens = 256 * len(images_to_add)  # 所有图片的tokens
                            total_required_tokens += template_tokens + image_tokens
                            
                            # 检查是否能添加所有内容
                            if current_tokens + total_required_tokens >= self.config.response_length:           # 无法添加所有内容，只添加response
                                # print(f"Cannot add all images. Required tokens: {total_required_tokens}, Current: {current_tokens}, Max: {self.config.response_length}. "
                                #             f"Only adding response.")
                                samples_info[index]['sequence'] += result['response']
                                samples_info[index]['total_response_tokens'] += response_tokens
                                samples_info[index]['stop'] = True
                                samples_info[index]['finish_reason'] = "max_tokens"
                            else:
                                # 可以添加所有内容
                                samples_info[index]['sequence'] += result['response'] + template_str
                                samples_info[index]['multi_modal_data']['image'].extend(images_to_add)
                                samples_info[index]['total_response_tokens'] = current_tokens + total_required_tokens
                                
                            # 更新其他信息
                            samples_info[index]['bbox_list'] = result['bbox_list']
                            samples_info[index]["movement_list"] = result['movement_list']

                            # ----------- old -----------
                            # pad_prompt = ""
                            # current_tokens = samples_info[index]['total_response_tokens']
                            # response_tokens = len(result['response_ids'])
                            # for tmp_image_idx, tmp_image in enumerate(result["image"]):
                            #     width, height = fetch_image({"image": tmp_image}).size

                            #     if current_image_count + tmp_image_idx + 1>=MAX_IMAGES:
                            #         # 达到上限，使用final_prompt
                            #         pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + final_prompt.format(
                            #             current_image_idx=current_image_idx + tmp_image_idx,
                            #             width=width,
                            #             height=height,
                            #         )
                            #         samples_info[index]['multi_modal_data']['image'].append(tmp_image)
                            #         break
                            #     else:
                            #         if tmp_image_idx <= len(result["image"]) - 2:
                            #             pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>The index of the given image is {current_image_idx+tmp_image_idx} (width: {width}, height: {height}).\n"
                            #         else:
                            #             # if num_llm_calls_available>0:
                            #             if num_llm_calls_available_list[index] > 0:
                            #                 pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + intermediate_prompt.format(
                            #                     current_image_idx=current_image_idx + tmp_image_idx,
                            #                     width=width,
                            #                     height=height,
                            #                 )
                            #             else:
                            #                 pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + final_prompt.format(
                            #                     current_image_idx=current_image_idx + tmp_image_idx ,
                            #                     width=width,
                            #                     height=height,
                            #                 )
                            #         samples_info[index]['multi_modal_data']['image'].append(tmp_image)
                            # samples_info[index]['sequence'] += result['response'] + intermediate_template.format(prompt="", pad=pad_prompt)
                            # print(samples_info[index]['sequence'])
                        else:
                            current_image_idx = current_image_count + 1
                            width, height = fetch_image({"image": result["image"][0]}).size
                            # 检查是否达到图片数量上限
                            if current_image_count + 1>= MAX_IMAGES:
                                # 达到上限，使用final_prompt
                                prompt = final_prompt.format(
                                    current_image_idx=current_image_idx,
                                    width=width,
                                    height=height,
                                )
                            else:
                                # 未达到上限，根据调用次数决定使用哪个prompt
                                prompt = (intermediate_prompt if num_llm_calls_available > 0 else final_prompt).format(
                                    current_image_idx=current_image_idx,
                                    width=width,
                                    height=height,
                                )
                            # 构建完整的新增内容
                            new_template = intermediate_template.format(prompt=prompt, pad="<|vision_start|><|image_pad|><|vision_end|>")

                            # 计算添加新内容后的总token数
                            current_tokens = samples_info[index]['total_response_tokens']
                            response_tokens = len(result['response_ids'])
                            template_tokens = len(self.tokenizer.encode(new_template, add_special_tokens=False))
                            image_tokens = 256  # 估计每张图片占用的tokens
                            potential_total_tokens = current_tokens + response_tokens + template_tokens + image_tokens
                            if potential_total_tokens >= self.config.response_length:                            # 检查是否会超出限制
                                samples_info[index]['sequence'] += result['response']
                                samples_info[index]['total_response_tokens'] += response_tokens
                                samples_info[index]['stop'] = True
                                samples_info[index]['finish_reason'] = "max_tokens"
                            else:                                                                               # 可以添加完整的新内容
                                samples_info[index]['sequence'] += result['response'] + new_template
                                samples_info[index]['multi_modal_data']['image'].append(result['image'][0])
                                samples_info[index]['total_response_tokens'] = potential_total_tokens
                                # samples_info[index]['sequence'] += result['response'] + intermediate_template.format(
                                #     prompt=prompt,
                                #     pad="<|vision_start|><|image_pad|><|vision_end|>"
                                # )
                                # samples_info[index]['multi_modal_data']['image'].append(result['image'][0])
                        samples_info[index]['bbox_list'] = result['bbox_list']
                        samples_info[index]["movement_list"] = result['movement_list']
                    else:
                        samples_info[index]['sequence'] += result['response']

        for i, line in enumerate(samples_info):
            if samples_info[i]['finish_reason']!='length': 
                samples_info[i]['sequence']+=self.tokenizer.eos_token                # add end of sentence
                samples_info[i]['response']+=self.tokenizer.eos_token                # add end of sentence
                
        responses = []      # mutli-turn response beginning from first prompt
        sequences = []      # complete multi-turn conversations
        image_inputs = []
        for idx, sample_info in enumerate(samples_info):
            responses.append(sample_info['response'])
            sequences.append(sample_info['sequence'])
            image_inputs.append(sample_info['multi_modal_data']['image'])

        if save_dir:
            all_sample_dir = save_samples_info(samples_info[:1], save_dir)
        return responses, sequences, image_inputs

    def _mask(self, ):
        pass

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (rollout_bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data, grid_size in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), non_tensor_batch.pop("grid_size")             # origin: pop
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data, "grid_size": grid_size})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            responses, sequences, image_inputs = self._multi_turn_generate(vllm_inputs=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False, save_dir=None)       # save_dir="./tmp_trace/20250512"

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)           # repeat tensor or list at dim=0
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            
            
        non_tensor_batch["raw_prompt_ids"] =  [self.tokenizer.encode(sequence, add_special_tokens=False)[:self.config.prompt_length+self.config.response_length] for sequence in sequences] # raw prompt+response ids
        valid_prompt_len = torch.sum(attention_mask, dim=-1)
        response_ids = []
        response_mask = []
        response_position_ids = []
        model_inputs = []
        multi_turn_mask = []
        # sequence_ids = []
        for idx, prompt_len in enumerate(valid_prompt_len):
            inputs = self.processor(text=sequences[idx], 
                                    images=image_inputs[idx], 
                                    add_special_tokens=False,              # whether add special tokens like <|im_start|><||>
                                    # padding='max_length',                   # ['longest', 'max_length', 'do_not_pad']
                                    # max_length=self.config.prompt_length + self.config.response_length,
                                    return_tensors="pt")                    # for transformers
            new_position_ids = get_rope_index(
                self.processor,
                input_ids=inputs['input_ids'][0],
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs['attention_mask'][0],
            )  # (3, seq_length)
            assert torch.sum(input_ids[idx][-prompt_len:].cpu() == inputs['input_ids'][0][:prompt_len].cpu()) == prompt_len, \
                f"Input IDs mismatch at batch index {idx}"
            
            assert torch.sum(attention_mask[idx][-prompt_len:].cpu() == inputs['attention_mask'][0][:prompt_len].cpu()) == prompt_len, \
                f"Attention mask mismatch at batch index {idx}"
            assert torch.sum(position_ids[idx, :, -prompt_len:].cpu()== new_position_ids[: ,:prompt_len].cpu()) == prompt_len * 3, \
                f"Attention mask mismatch at batch index {idx}"

            assert torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151652) == len(image_inputs[idx]), \
                f"Number of <|vision_start|> {torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151652)} != Image Inputs {len(image_inputs[idx])}"
            assert torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151653) == len(image_inputs[idx]), \
                f"Number of <|vision_end|> {torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151653)} != Image Inputs {len(image_inputs[idx])}"
            # print(f"Number of <|vision_start|> {sum([x==151652 for x in non_tensor_batch['raw_prompt_ids'][idx]])}, Number of <|vision_end|> {sum([x==151653 for x in non_tensor_batch['raw_prompt_ids'][idx]])},  Image Inputs {len(image_inputs[idx])}")
            
            response_ids.append(inputs['input_ids'][0][prompt_len: prompt_len+self.config.response_length])
            response_mask.append(inputs['attention_mask'][0][prompt_len: prompt_len+self.config.response_length])
            pad_position_ids = VF.pad_sequence_to_length(new_position_ids[:, prompt_len: prompt_len + self.config.response_length], max_seq_len=self.config.response_length, pad_token_id=0, left_pad=False).to(input_ids.device)        # (3, max_length)
            response_position_ids.append(pad_position_ids)
            tmp_multi_turn_mask = self._get_multi_turn_mask(inputs['input_ids'][0][prompt_len: prompt_len+self.config.response_length])
            multi_turn_mask.append(tmp_multi_turn_mask)

            inputs.pop('input_ids')
            inputs.pop('attention_mask')
            model_inputs.append(dict(inputs))               # convert transformers.feature_extraction_utils.BatchFeature to dict

        print("Response Ids: ", [len(x) for x in response_ids])
        for idx, x in enumerate(response_ids):
            if len(x) > self.config.response_length:
                print(">response_length X:", x)
                print("Sequence: ", sequences[idx])
                print("\n")
        response_ids = VF.pad_2d_list_to_length(response_ids, self.pad_token_id, max_length=self.config.response_length).to(input_ids.device)                      # (b * n, max_length)
        non_tensor_batch["multi_modal_inputs"] = model_inputs

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)

        # # prompt: left pad + response: right pad
        # # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        # response_position_ids = position_ids[..., -1:] + delta_position_id
        response_position_ids = torch.stack(response_position_ids, dim=0).to(input_ids.device)                    # (b*n, 3, max_length)
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.pad_2d_list_to_length(response_mask, 0, max_length=self.config.response_length).to(input_ids.device)                      # (b * n, max_length)
        multi_turn_mask = VF.pad_2d_list_to_length(multi_turn_mask, 0, max_length=self.config.response_length).to(input_ids.device)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        valid_lengths = torch.sum(attention_mask, dim=1)  # 按行求和
        max_valid_length = torch.max(valid_lengths).cpu()   
        min_valid_length = torch.min(valid_lengths).cpu()
        avg_valid_length = torch.mean(valid_lengths.float()).cpu()
        image_pad_counts = torch.sum(sequence_ids == 151655, dim=1)  # 按行统计
        max_image_pads = torch.max(image_pad_counts).cpu()
        min_image_pads = torch.min(image_pad_counts).cpu()
        avg_image_pads = torch.mean(image_pad_counts.float()).cpu()

        print(f"Size of prompt_ids: {input_ids.size()}")
        print(f"Size of response_ids: {response_ids.size()}")
        print(f"Size of sequence_ids: {sequence_ids.size()}")
        print(f"Valid Length - Max: {max_valid_length}, Min: {min_valid_length}, Avg: {avg_valid_length:.2f}")
        print(f"Image_pad Number - Max: {max_image_pads}, Min: {min_image_pads}, Avg: {avg_image_pads:.2f}")

        batch = TensorDict(
            {
                "prompts": input_ids,                   # origin prompt ids
                "responses": response_ids,              # new response ids
                "input_ids": sequence_ids,              # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,                             # new response mask
                "position_ids": position_ids,
                "multi_turn_mask": multi_turn_mask,
            },
            batch_size=batch_size,
        )
        for key, value in non_tensor_batch.items():
            if isinstance(value, np.ndarray) is False:
                non_tensor_batch[key] = np.array(value, dtype=object)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


