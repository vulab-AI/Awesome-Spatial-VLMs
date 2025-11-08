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

import re
import os
import ast
from typing import Dict
from rouge_score import rouge_scorer
import torch


def parse_bbox_and_movement(response):
    def parse_json(json_output):
        # Parsing out the markdown fencing
        json_output_list = []
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            # print(i, line, line.strip()=="```json")
            if line.strip() == "```json":
                tmp = "\n".join(lines[i+1:])  # Remove everything before "```json"
                tmp = tmp.split("```")[0]  # Remove everything after the closing "```"
                json_output_list.append(tmp)
        return json_output_list

    def safe_eval(item):
        try:
            # 使用 ast.literal_eval 代替 eval，以安全地评估字符串表达式
            return ast.literal_eval(item)
        except (ValueError, SyntaxError) as e:
            # 如果解析失败，记录错误信息并返回 None
            print(f"Failed to evaluate item: {item}. Error: {e}")
            return None

    parsed_list = parse_json(response)
    parsed_list = [safe_eval(item) for item in parsed_list]
    fail_parse_response_list = []
    bbox_list, movement_list, not_bbox_movement_list  = [], [], []
    for item_list in parsed_list:
        if item_list is None:
            fail_parse_response_list.append(item_list)
            continue

        for item in item_list:
            if "bbox_2d" in item and "label" in item:
                bbox_list.append(item)
            elif "start_point_2d" in item and "end_point_2d" in item and "label" in item:
                movement_list.append(item)
            else:
                not_bbox_movement_list.append(item)
    return bbox_list, movement_list, fail_parse_response_list, not_bbox_movement_list


def format_reward(conversations):
    reward = 0.0

    # 1. assistant perform at least 2 actions and 1 answer
    # if (len(conversations)-1)//2 >= 3:                         #  system, user, assistant, user, assistant...      
    #     reward += 1
    # else:
    #     return 0.0

    # 2. whether follow format requirement <think></think> <answer></answer> or <think></think>Action:...
    format_flag = True
    total_content = ""
    for i in range(2, len(conversations), 2):
        content = conversations[i]['content'] 
        pattern_action = r'<think>(.*?)</think>\s*Action:\s*(.*?)(?=\n|$)'
        match_action = re.fullmatch(pattern_action, content, re.DOTALL)
        
        # either match answer in the last round or match action not in the last round
        if match_action is None \
            or (i == len(conversations)-1 and ("<answer>" not in content or "</answer>" not in content)) \
            or (i != len(conversations)-1 and ("<answer>" in content or "</answer>" in content)):
            format_flag = False
            break
        total_content += content
    # whether output action, bbox and movement follow the instruction
    if format_flag is True:
        bbox_list, movement_list, fail_parse_response_list, not_bbox_movement_list = parse_bbox_and_movement(total_content)
        if len(fail_parse_response_list) > 0 or len(not_bbox_movement_list) > 0:
            format_flag = False
        
    if format_flag is True:
        reward = reward + 1
    else:
        reflection_reward=0.0
        return reward / 2.0, reflection_reward
    
    # 3. reward lengthy response
    # if (len(conversations)-1)//2 >= 5:                         
    #     reward += 1

    # 4.whether change original bbox
    bbox_dict = {} 
    bbox_modify_flag = False
    reflection_reward = 0
    # 遍历所有bbox，检查是否有相同label但不同的bbox_2d，表示坐标发生了变动
    for bbox in bbox_list:
        # 确保bbox_2d是tuple类型
        try:
            bbox_2d = tuple(map(float, bbox['bbox_2d']))  # 转换为float并转为tuple
            bbox_label = bbox['label']

            if bbox_label not in bbox_dict:
                # 初次出现该label，记录bbox
                bbox_dict[bbox_label] = set()
                bbox_dict[bbox_label].add(bbox_2d)
            else:
                # 如果相同label对应多个不同的bbox_2d，说明坐标有改动
                if bbox_2d not in bbox_dict[bbox_label]:
                    bbox_modify_flag = True
                    bbox_dict[bbox_label].add(bbox_2d)
                    # print(f"Label '{bbox_label}' has different bbox_2d: {bbox_dict[bbox_label]}")
        except (TypeError, ValueError) as e:
            print(f"Error processing bbox: {bbox}, Error: {str(e)}")
            continue
    if bbox_modify_flag is True:
        # reward = reward + 1
        reflection_reward += 1
        
    # reward = reward / 2.0                 # normalize
    return reward, reflection_reward


def parse_dialog(serialized_content):
    # 分割对话内容
    segments = re.split(r'<\|im_start\|>|<\|im_end\|>', serialized_content)
    segments = [s for s in segments if s]  # 移除空字符串，但不strip
    
    conversations = []
    
    # 系统提示的处理
    system_content = None
    if segments[0].startswith('system'):
        system_content = segments[0].replace('system\n\n', '', 1)  # 只替换第一次出现
        segments = segments[1:]
    
    # 初始化对话列表
    if system_content:
        conversations.append({
            "role": "system",
            "content": system_content
        })
    
    # 处理用户和助手的对话
    for segment in segments:
        if segment.startswith('user'):
            # 提取图像标记和文本
            has_vision = '<|vision_start|><|image_pad|><|vision_end|>' in segment
            text = segment.replace('user\n', '', 1)  # 只替换第一次出现
            text = text.replace('<|vision_start|><|image_pad|><|vision_end|>\n', '', 1)
            
            content = []
            if has_vision:
                content.append({
                    "type": "image",
                    "image": "image_path",
                    "nframes": "args.max_frames",
                    # "max_pixels": 256*28*28
                })
            content.append({
                "type": "text",
                "text": text
            })
            
            conversations.append({
                "role": "user",
                "content": content
            })
        elif segment.startswith('assistant'):
            text = segment.replace('assistant\n', '', 1)  # 只替换第一次出现
            conversations.append({
                "role": "assistant",
                "content": text
            })
    return conversations


def accuracy_reward(output, ground_truth, question_type):

    def extract_answer(text):
        """提取答案，确保返回字符串"""
        try:
           
            if text is None:
                return ""
            if not isinstance(text, (str, bytes)):
                text = str(text)

            pattern = r'<answer>\s*(.*?)\s*</answer>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text.strip()
        except Exception as e:
            print(f"Error in extract_answer: {str(e)}, Input type: {type(text)}")
            return ""

    def normalize_number(num_str):
        try:
            if not isinstance(num_str, str):
                num_str = str(num_str)
                
            number_pattern = r'([-+]?\d*\.?\d+)'
            match = re.search(number_pattern, num_str)
            
            if match:
                num_str = match.group(1).rstrip('.')
                return float(num_str)
            else:
                print(f"No valid number found in '{num_str}'")
                return None
        except Exception as e:
            print(f"Error processing '{num_str}': {e}")
            return None


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()

    try:
        output_ans = extract_answer(output)
        gt_ans = extract_answer(ground_truth)
        if question_type == "multiple choice":
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type in  ["regression", 'numerical']:
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                reward = 0.0
            else:
                reward = mean_relative_accuracy(out_number, gt_number)
        else:
            reward = 0.0
    except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}. Output: {output}, GroundTruth: {ground_truth}")
            reward = 0.0
    return reward


def gcot_compute_score(predict_str: str, ground_truth: str, question_type: str, data_type: str) -> Dict[str, float]:
    """
    Computes the overall reward, as well as the format and accuracy rewards.
    
    Args:
        predict_str (str): The predicted response string.
        ground_truth (str): The correct answer string.
        
    Returns:
        Dict[str, float]: A dictionary containing the overall, format, and accuracy rewards.
    """
    conversations: list = parse_dialog(predict_str)
    format, reflection = format_reward(conversations)
    accuracy = accuracy_reward(conversations[-1]['content'], ground_truth, question_type)
    print(f"Question Type: {question_type}, Format: {format}, Accuracy: {accuracy}")
    return {
        "overall": 0.5 * accuracy + 0.5 * format if accuracy > 0.0 else accuracy,
        "format": format,
        "accuracy": accuracy,
        "reflection": reflection,
        f"{data_type}_accuracy": accuracy,
        f"{data_type}_format": format,
    }
