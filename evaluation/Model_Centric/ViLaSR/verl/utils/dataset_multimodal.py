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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_vl_utils import fetch_image

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json

SYSTEM_PROMPT = """### Guidance:
You are a spatial reasoning assistant with access to two powerful visualization tools.
Your task is to break down complex spatial problems and iteratively refine your solution through visualization feedback.

### Available tools:
You can use the following two tools to visualize. After each tool usage, you must wait for and analyze the visualization feedback before proceeding.

1. **Object Mapper**
- Purpose: Identifies and maps key items in the space
- Input format: JSON
```json
[{{
    "index": i, # Image index
    "bbox_2d": [x1, y1, x2, y2],
    "label": "object name/description"
}}]
```
- Output: Generates bounding boxes for visual inspection of the i-th image

2. **Path Tracer**
- Purpose: Plots movement or connections between points
- Input format: JSON
```json
[{{
    "index": i, # Image index
    "start_point_2d": [x1, y1],
    "end_point_2d": [x2, y2],
    "label": "trace_description"
}}]
```
- Output: Generates visual paths for verification of the i-th image

### Required Output Format:
For each reasoning step, you must structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: [Object Mapper/Path Tracer]
```json
[JSON format coordinates]
```

After your reasoning and iteratively refine your solution through visualization feedback, you should arrive at a final answer and structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: Answer
<answer> [Your final answer] </answer>

### Please NOTE the following reasoning techniques:
1. Initial Analysis
   - Break down the spatial problem
   - Plan your approach

2. Iterative Reasoning for Each Step
   - Choose appropriate tool
   - Provide absolute coordinates in JSON format (The top-left corner of the image is (0, 0) and the bottom-right corner is ({width}, {height}))
   - Observe the visualization output
   - Reflect on the visualization:
     * Is the placement/path accurate?
     * Does it align with your reasoning?
     * What adjustments are needed?
   - Backtrack and Adjust:
     * If errors found, backtrack to previous step to modify actions or decisions as needed"""

PROMPT_TEMPLATE = """
### Question:
{question}

Begin your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:
"""
TYPE_TEMPLATE = {
    "multiple choice":  '\nAnswer with the option\'s letter from the given choices directly.',  #  " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    # "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": '', # " Please provide your text answer within the <answer> </answer> tags.",
    "regression":  '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).', # " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    "numerical": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).'
    # "subj": ''
}

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = SYSTEM_PROMPT # system_prompt
        self.prompt_template = PROMPT_TEMPLATE
        self.type_template = TYPE_TEMPLATE
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        # if os.path.isdir(data_path):
        #     self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        # elif os.path.isfile(data_path):
        #     self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        # else:  # remote dataset
        #     self.dataset = load_dataset(data_path, split=data_split)
        self.dataset = [json.loads(x) for x in open(data_path, "r")]
        # self.dataset = HFDataset.from_list(self.dataset)

    def __len__(self):
        return len(self.dataset)

    
    def process_image(self, image_path):
        """处理图像"""
        return fetch_image({"image": image_path, "max_pixels": self.max_pixels})


    def construct_video_prompt(self, item, size):
        """构造视频数据的prompt"""
        # 基础提示语
        # prompt = ("These are frames from a video, numbered from 1 to 16 in sequence. "
        #          "That is, the index of each image is 1, 2, 3, ..., 16.\n\n"
        #          f"Answer the question with appropriate tools:\n{item[self.prompt_key]}\n\n"
        #          "The final answer should be a single word or phrase.")
        prompt = "These are frames from a video, numbered from 1 to 16 in sequence. That is, the index of each image is 1, 2, 3, ..., 16.\n\nAnswer the quesntion with appropriate tools:\n" \
                + item[self.prompt_key] 
        if item['question_type'] == 'multiple choice' and 'options' in item:
            item['options'] = [op.strip() for op in item['options']]
            prompt = prompt + '\n' +  '\n'.join(item['options'])
        prompt += TYPE_TEMPLATE[item['question_type'].lower()]

        width, height = size            # 获取图像尺寸
        # 构建消息列表
        image_messages = []
        for idx, image_path in enumerate(item[self.image_key]):
            image_messages.extend([
                {
                    "type": "image",
                    "image": image_path,
                    # "nframes": self.max_frames,
                    # "max_pixels": self.max_pixels
                },
                {
                    "type": "text",
                    "text": f"The index of the given image is {idx+1} (width: {width}, height: {height}).\n"
                }
            ])
            
        image_messages.append({
            "type": "text",
            "text": self.prompt_template.format(question=prompt)
        })
        
        return [
            {
                "role": "system",
                "content": self.system_prompt.format(width=width, height=height)
            },
            {
                "role": "user",
                "content": image_messages
            }
        ]
    
    
    def construct_maze_vqa_prompt(self, item, size):
        """构造迷宫或VQA数据的prompt"""
        # if item['data_type'] == "maze":
        #     prompt = (f"{item[self.prompt_key]}\nThe index of the given image is 1."
        #              f"{self.type_template['multiple choice']}")
        # else:
        #     prompt = f"{item[self.prompt_key]}\nThe index of the given image is 1."
        prompt =  item[self.prompt_key]
        if item['question_type'] == 'multiple choice' and 'options' in item:
            item['options'] = [op.strip() for op in item['options']]
            prompt = prompt + '\n' +  '\n'.join(item['options'])
        prompt =  prompt + '\nThe index of the given image is 1.' + TYPE_TEMPLATE[item['question_type'].lower()]
            
        width, height = size
        return [
            {
                "role": "system",
                "content": self.system_prompt.format(width=width, height=height)
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item[self.image_key][0],
                        # "nframes": self.max_frames,
                        # "grid_size": item["grid_size"] if "grid_size" in item else None,      # grid size not implemented
                        # "max_pixels": self.max_pixels
                    },
                    {
                        "type": "text",
                        "text": self.prompt_template.format(question=prompt)
                    }
                ]
            }
        ]
    
    def __getitem__(self, index):
        """获取数据项"""
        row_dict: dict = self.dataset[index].copy()
        data_type = row_dict['data_type']
        # print(index, row_dict.keys(), self.image_key in row_dict)
        # if self.image_key not in row_dict:
        #     print(row_dict)
        
        # if data_type == 'video':
        images = [fetch_image({"image": image_path, "max_pixels": self.max_pixels}) for image_path in row_dict[self.image_key] ]
        # else:
        #     images = [fetch_image({"image": row_dict[self.image_key], "max_pixels": self.max_pixels})  ]
        # construct messages
        if data_type in ["video", "spatial_r1"]:
            messages = self.construct_video_prompt(row_dict, images[0].size)
        elif data_type in ["maze", "vqa"]:
            messages = self.construct_maze_vqa_prompt(row_dict, images[0].size)
        else:
            raise ValueError(f"Unknown data_type: {row_dict['data_type']}")
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_data"] = {"image": images}
        row_dict["multi_modal_inputs"] = dict(model_inputs)             # num of <|image_pad|>: h*w //(merge_size*2)
        row_dict.pop(self.image_key)
        
        # 处理position_ids
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        
        return row_dict

from transformers import AutoTokenizer, AutoProcessor
def test_rlhf_dataset():
    # 1. 初始化必要的组件
    tokenizer = AutoTokenizer.from_pretrained("/ossfs/workspace/nas/public_ckpts/Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("/ossfs/workspace/nas/public_ckpts/Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    
    # 2. 创建数据集实例
    dataset = RLHFDataset(
        data_path="/ossfs/workspace/nas/wujunfei/code/gvsr_o3/data/rl_data/merged_data_49k_train.jsonl",
        tokenizer=tokenizer,
        processor=processor,
        prompt_key = "question",
        answer_key = "answer",
        image_key = "image_path",
        max_prompt_length=5120,
        max_pixels=256*28*28,  # 1M pixels
        min_pixels=16384     # 64K pixels
    )
    
    # 3. 基本信息测试
    print(f"Dataset size: {len(dataset)}")
    
    # 定义要检查的关键字段
    essential_fields = [
        "input_ids", "attention_mask", "position_ids", 
        "multi_modal_data", "multi_modal_inputs",
        "ground_truth", "data_type"
    ]
    
    # 记录错误样本
    error_samples = []
    
    # 遍历所有样本
    for idx in range(len(dataset)):
        try:
            # 获取样本
            sample = dataset[idx]
            # print(sample)
            
            # 检查关键字段
            missing_fields = [field for field in essential_fields if field not in sample]
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")
            
            # 检查张量形状
            assert sample['input_ids'].shape == sample['attention_mask'].shape, \
                "input_ids and attention_mask shapes don't match"
            
            # 检查图像数据
            images = sample['multi_modal_data']['image']
            if isinstance(images, list):
                for i, img in enumerate(images):
                    assert hasattr(img, 'size'), f"Invalid image at index {i}"
                    assert img.size[0] * img.size[1] <= 256*28*28, f"Image {i} too large: {img.size}"
                    assert img.size[0] * img.size[1] >= 16384, f"Image {i} too small: {img.size}"
            else:
                assert hasattr(images, 'size'), "Invalid image"
                assert images.size[0] * images.size[1] <= 256*28*28, f"Image too large: {images.size}"
                assert images.size[0] * images.size[1] >= 16384, f"Image too small: {images.size}"
            
            # 每100个样本打印一次进度和内存使用情况
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples")
                print(f"Current sample shapes:")
                print(f"- input_ids: {sample['input_ids'].shape}")
                print(f"- attention_mask: {sample['attention_mask'].shape}")
                print(f"- position_ids: {sample['position_ids'].shape}")
                print("---")
                
        except Exception as e:
            error_msg = f"Error in sample {idx}: {str(e)}"
            print(error_msg)
            error_samples.append((idx, error_msg))
            
    # 打印最终统计信息
    print("\nValidation completed!")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Successful samples: {len(dataset) - len(error_samples)}")
    print(f"Failed samples: {len(error_samples)}")
    
    # 如果有错误样本，打印详细信息
    if error_samples:
        print("\nError details:")
        for idx, error in error_samples:
            print(f"Sample {idx}: {error}")
    
    # 打印一个成功样本的详细信息
    if len(dataset) > 0 and len(error_samples) < len(dataset):
        # 找第一个成功的样本
        for idx in range(len(dataset)):
            if idx not in [x[0] for x in error_samples]:
                sample = dataset[idx]
                print("\nExample of successful sample:")
                print(f"Data type: {sample['data_type']}")
                print(f"Input IDs shape: {sample['input_ids'].shape}")
                print(f"Attention mask shape: {sample['attention_mask'].shape}")
                print(f"Position IDs shape: {sample['position_ids'].shape}")
                print(f"Ground truth: {sample['ground_truth']}")
                
                images = sample['multi_modal_data']['image']
                if isinstance(images, list):
                    print(f"Number of images: {len(images)}")
                    for i, img in enumerate(images):
                        print(f"Image {i} size: {img.size}")
                else:
                    print(f"Image size: {images.size}")
                break

if __name__ == "__main__":
    test_rlhf_dataset()
