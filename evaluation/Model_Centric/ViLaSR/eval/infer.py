import argparse
import traceback
import random
import re
import copy
import torch
import os
import json
from tqdm import tqdm
import pdb
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info, fetch_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.edit_image import merge_bbox_movement, parse_bbox_and_movement, plot_movement, plot_bounding_boxes
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time

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
#  (You can only edit ONE image at each step)
PROMPT_TEMPLATE = """
### Question:
{question}

Begin your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:
"""

BSZ=10                      # 50 reduce it if GPU OOM
MAX_IMAGES=45
SUBIMAGE_PATTERN = r".*\#\#\#\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
TYPE_TEMPLATE = {
    # ----------- same as cold start -----------
    "multiple choice":  '\nAnswer with the option\'s letter from the given choices directly.',  #  " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "free-form": '', # " Please provide your text answer within the <answer> </answer> tags.",
    "regression":  '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).', # " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    "numerical": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).',      # same as regression  
    "vci": "",                                                                                          # for spar-bench
}

from dataclasses import dataclass
@dataclass
class ProcessData:
    index: int
    response: str
    mm_data: Dict
    bbox_list_origin: Dict
    movement_list_origin: Dict
    finish_reason: str
    is_finished: bool
    grid_size: int

def calculate_grid_centers(image_size=616, grid_size=5):
    """for maze data"""

    # matplotlib default margins
    margin_left = int(image_size * 0.125)    
    margin_right = int(image_size * 0.1)     
    margin_bottom = int(image_size * 0.11)   
    margin_top = int(image_size * 0.12)      
    
    usable_width = image_size - (margin_left + margin_right)
    usable_height = image_size - (margin_top + margin_bottom)
    
    cell_width = usable_width / grid_size
    cell_height = usable_height / grid_size
    
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

        try:
            allindex = {}
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

        if len(allindex) == 0:
            finish_reason = "ToolError"
        elif len(data.mm_data['image']) >= MAX_IMAGES+1:
            finish_reason = "TooManyImages"


        if finish_reason is not None:
            return {
                'index': data.index,
                'processed_image_idx': [None],
                'image': [data.mm_data['image'][0].copy()],
                'response': data.response,
                'finish_reason': finish_reason,
                'bbox_list': bbox_list,
                'movement_list': movement_list,
                'is_finished': True,
            }
        for cnt, tmp_index in enumerate(allindex):
            bbox_list_new, movement_list_new = allindex[tmp_index]["bbox_list"], allindex[tmp_index]["movement_list"]
            image_index_new = current_image_index + cnt
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
                    'finish_reason': "ToolError",
                    'bbox_list': bbox_list,
                    'movement_list': movement_list,
                    'is_finished': True,
                }
            image = data.mm_data['image'][image_index].copy()
            assert isinstance(image, Image.Image)
            input_width, input_height = image.size

            # draw bbox and lines
            plot_bounding_boxes(image, bbox_list[image_index_new], input_height=input_height, input_width=input_width)
            plot_movement(image, movement_list[image_index_new], input_height=input_height, input_width=input_width)
            
            image_list.append(image)

        return {
            'index': data.index,
            'processed_image_idx': image_index_list,
            'image': image_list,
            'response': data.response,
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
        sample_dir = get_unique_dir(save_dir, f'sample')
        os.makedirs(sample_dir, exist_ok=True)
        all_sample_dir.append(sample_dir)

        text_data = {
            'prompt': sample['prompt'],
            'sequence': sample['sequence'],
            'response': sample['response'],
            'finish_reason': sample['finish_reason'],
            'execution_pass': sample['execution_pass']
        }
        
        with open(os.path.join(sample_dir, 'text_data.json'), 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        # save images
        if 'multi_modal_data' in sample and 'image' in sample['multi_modal_data']:
            images_dir = os.path.join(sample_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            for img_idx, img in enumerate(sample['multi_modal_data']['image']):
                if isinstance(img, Image.Image):
                    img_path = os.path.join(images_dir, f'image_{img_idx}.png')
                    img.save(img_path)
    return all_sample_dir

def multi_turn_generate(inference_engine, tokenizer, vllm_inputs=None, sampling_params=None, prompt_token_ids=None,  use_tqdm=False, save_dir=None, max_num_steps=10):
    def _get_prompts_and_indices(samples_info):
        prompts, multi_modal_data, indices=[], [], []
        for index, info in enumerate(samples_info):
            if not info['stop'] and len(info['multi_modal_data']['image']) <= MAX_IMAGES:
                prompts.append(info['sequence'])
                multi_modal_data.append(info['multi_modal_data'])
                indices.append(info['index'])
        return prompts, multi_modal_data, indices

    sampling_params=copy.deepcopy(sampling_params)
    new_vllm_inputs = []
    for single_vllm_input in vllm_inputs:
        prompt = tokenizer.decode(single_vllm_input['prompt_token_ids'], skip_special_tokens=False)
        new_vllm_inputs.extend([{
            "prompt": prompt,
            "multi_modal_data": single_vllm_input['multi_modal_data'],
            "grid_size": single_vllm_input['grid_size'],
        }   for _ in range(sampling_params.n)])
        
    sampling_params.n=1
    sampling_params.detokenize=True             # True convert ids to text
    samples_info = []
    for index, item in enumerate(new_vllm_inputs):
        
        processed_image = [fetch_image({'image': origin_image}) for origin_image in item['multi_modal_data']['image']]
        sample_info = {
            "prompt": item["prompt"],
            "sequence": item["prompt"],
            "multi_modal_data": {"image": processed_image},
            "response": "",
            "stop": False,
            "finish_reason": None,
            "processed_image_idx": [],
            "index": index,
            "mask_info": [],
            "execution_pass": 0,
            "bbox_list": {img_idx: [] for img_idx in range(len(processed_image))},        # save bbox_list for each image index 
            "movement_list": {img_idx: [] for img_idx in range(len(processed_image))},     # save movement_list for each image index
            "grid_size": item['grid_size'],
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
"""                 # connect multi-turn conversations
    num_llm_calls_available = max_num_steps - 1
    while num_llm_calls_available >= 0:
        num_llm_calls_available-=1
        input_prompts, multi_modal_data, indices=_get_prompts_and_indices(samples_info)  # _get_prompts_and_indices
        # print("input_prompts:", input_prompts[0])
        input_prompts = [{
            'prompt_token_ids': tokenizer.encode(prompt, add_special_tokens=False)[:],
            'multi_modal_data': mm_data                             # {'image', list()}
            } for prompt, mm_data in zip(input_prompts, multi_modal_data)]
        outputs = inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)

        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        responses=[x.outputs[0].text for x in sorted_outputs]
        finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]  # "stop", "length"
        stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]      # None: have EOS
        if num_llm_calls_available==-1:
            for i ,index in enumerate(indices):
                samples_info[index]['response']+=responses[i]
                samples_info[index]['sequence']+=responses[i]
                samples_info[index]['stop']=True
                samples_info[index]['finish_reason']=finish_reason[i]
            break

        def _is_finished(finish_reason, stop_reason, response):
            if finish_reason=='stop' and stop_reason==None and "<answer>" in response and "</answer>" in response: 
                return True
            if finish_reason=='length':
                return True
            if finish_reason=='rule':
                return True
            return False
        
        # breakpoint()
        is_finished=[_is_finished(finish_reason[i], stop_reason[i], responses[i]) for i in range(len(finish_reason))]
        # check if all samples are finished
        if all([x for x in is_finished]): 
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
                mm_data=samples_info[index]['multi_modal_data'],
                bbox_list_origin=samples_info[index]["bbox_list"],
                movement_list_origin=samples_info[index]["movement_list"],
                finish_reason=finish_reason[i],
                is_finished=is_finished[i],                      # if is_finished == True, stop reasoning
                grid_size=samples_info[index]['grid_size'],
            )  for i, index in enumerate(indices)] 
        with ThreadPoolExecutor(max_workers=max(min(len(indices), os.cpu_count()//2, 64), 1) ) as executor:
            results = list(executor.map(process_single_response, process_data_list))

        # update samples_info
        for result in results:
            if result is not None:
                index = result['index']
                samples_info[index]['response'] += result['response']
                samples_info[index]['stop'] = result['is_finished']
                samples_info[index]['finish_reason'] = result['finish_reason']
                samples_info[index]['processed_image_idx'].extend(result['processed_image_idx'])
                if result['is_finished'] is False:
                    current_image_count = len(samples_info[index]['multi_modal_data']['image'])  
                    if len(result["image"]) > 1:        # 处理多图片情况
                        current_image_idx = current_image_count + 1
                        pad_prompt = ""
                        for tmp_image_idx, tmp_image in enumerate(result["image"]):
                            width, height = fetch_image({"image": tmp_image}).size
                            if current_image_count + tmp_image_idx + 1>=MAX_IMAGES:
                                pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + final_prompt.format(
                                    current_image_idx=current_image_idx + tmp_image_idx,
                                    width=width,
                                    height=height,
                                )
                                samples_info[index]['multi_modal_data']['image'].append(tmp_image)
                                break
                            else:
                                if tmp_image_idx <= len(result["image"]) - 2:
                                    pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>The index of the given image is {current_image_idx+tmp_image_idx} (width: {width}, height: {height}).\n"
                                else:
                                    if num_llm_calls_available > 0:
                                        pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + intermediate_prompt.format(
                                            current_image_idx=current_image_idx + tmp_image_idx,
                                            width=width,
                                            height=height,
                                        )
                                    else:
                                        pad_prompt += f"<|vision_start|><|image_pad|><|vision_end|>" + final_prompt.format(
                                            current_image_idx=current_image_idx + tmp_image_idx,
                                            width=width,
                                            height=height,
                                        )
                                samples_info[index]['multi_modal_data']['image'].append(tmp_image)
                        
                        samples_info[index]['sequence'] += result['response'] + intermediate_template.format(prompt="", pad=pad_prompt)  
                    else:
                        current_image_idx = current_image_count + 1
                        width, height = fetch_image({"image": result["image"][0]}).size
                        if current_image_count + 1>= MAX_IMAGES:
                            # Maximum limit reached, switching to final_prompt
                            prompt = final_prompt.format(
                                current_image_idx=current_image_idx,
                                width=width,
                                height=height,
                            )
                        else:
                            
                            prompt = (intermediate_prompt if num_llm_calls_available > 0 else final_prompt).format(
                                current_image_idx=current_image_idx,
                                width=width,
                                height=height,
                            )
                        
                        samples_info[index]['sequence'] += result['response'] + intermediate_template.format(
                            prompt=prompt,
                            pad="<|vision_start|><|image_pad|><|vision_end|>"
                        )
                        samples_info[index]['multi_modal_data']['image'].append(result['image'][0])
                    
                    # 更新其他信息
                    samples_info[index]['bbox_list'] = result['bbox_list']
                    samples_info[index]["movement_list"] = result['movement_list']
                else:
                    samples_info[index]['sequence'] += result['response']

    for i, line in enumerate(samples_info):
        if samples_info[i]['finish_reason']!='length': 
            samples_info[i]['sequence']+=tokenizer.eos_token                # add end of sentence

    batch_sequences = [sample['sequence'] for sample in samples_info]
    if save_dir:
        all_sample_dir = save_samples_info(samples_info, save_dir)
        return batch_sequences, all_sample_dir
    return batch_sequences


def parse_dialog(serialized_content):
    # segement dialogue
    segments = re.split(r'<\|im_start\|>|<\|im_end\|>', serialized_content)
    segments = [s for s in segments if s]  
    
    conversations = []
    current_role = None
    current_content = []
    
    system_content = None
    if segments[0].startswith('system'):
        system_content = segments[0].replace('system\n\n', '', 1)  # only replace the first time
        segments = segments[1:]
    
    if system_content:
        conversations.append({
            "role": "system",
            "content": system_content
        })

    for segment in segments:
        if segment.startswith('user'):
            has_vision = '<|vision_start|><|image_pad|><|vision_end|>' in segment
            text = segment.replace('user\n', '', 1)  # only replace the first time
            # text = text.replace('<|vision_start|><|image_pad|><|vision_end|>\n', '', 1)               # keep <|vision_start|><|image_pad|><|vision_end|>
            
            content = []
            if has_vision:
                content.append({
                    "type": "image",
                    "image": "image_path",
                    "nframes": "args.max_frames",
                    "max_pixels": args.max_pixels
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
            text = segment.replace('assistant\n', '', 1)  # only replace the first time
            conversations.append({
                "role": "assistant",
                "content": text
            })
    
    return conversations



def eval_model(args):
    # Model
    model_path = args.model_path
    model_name = args.model_name

    print(f"Loading from {model_path}")
    llm = LLM(
            model=model_path, 
            dtype="bfloat16", 
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 62, "video": 10},
            gpu_memory_utilization=0.85,                   # default 0.9
            enable_prefix_caching=True                     # cache
          )
    processor = AutoProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=16384,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    file_path = args.input_file
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    st, ed = (len(data)*args.split)//args.all, (len(data)*(args.split+1))//args.all
    # st, ed = 10, 50
    print(f"{len(data)} lines found, generating from {st} to {ed}")
    print("Data: ", len(data))
    data = data[st:ed]
    messages = []
    for xidx, x in enumerate(data[:]):
        if args.dataset in ["vsi_bench"]:
            prompt = f"These are frames from a video, numbered from 1 to {args.max_frames} in sequence. That is, the index of each image is 1, 2, 3, ..., {args.max_frames}.\n\nAnswer the quesntion with appropriate tools:\n" + x['question'] # + '\n\nThe final answer should be a single word or phrase.'
            if x['problem_type'] == 'multiple choice' and 'options' in x:
                prompt = prompt + '\n' +  '\n'.join(x['options'])
            prompt = prompt + TYPE_TEMPLATE[x['problem_type'].lower()]
            width, height = fetch_image({"image": os.path.join(args.image_folder, x["image_path"][0]), "max_pixels": args.max_pixels}).size
            image_messages = []
            for image_idx, image_path in enumerate(x["image_path"]):
                image_messages.extend([
                    {
                        "type": "image",
                        "image": os.path.join(args.image_folder, image_path),
                        "nframes": args.max_frames,
                        "max_pixels": args.max_pixels
                    },
                    {
                        "type": "text",
                        "text": f"The index of the given image is {image_idx+1} (width: {width}, height: {height}).\n",
                    }
                ])
            image_messages.append({
                "type": "text",
                "text": PROMPT_TEMPLATE.format(question=prompt)
            })
            msg = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(width=width, height=height)
                },
                {
                    "role": "user",
                    "content": image_messages,
                }
            ]       
        elif args.dataset in ["maze", "SpatialEval_spatialreal",]:
            prompt =  x["question"]
            if x['problem_type'] == 'multiple choice' and 'options' in x:
                prompt = prompt + '\n' +  '\n'.join(x['options'])
            prompt =  prompt + '\nThe index of the given image is 1.' + TYPE_TEMPLATE[x['problem_type'].lower()]
            width, height = fetch_image({"image": os.path.join(args.image_folder, x["image_path"][0]), "max_pixels": args.max_pixels}).size
            msg = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(width=width, height=height)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.join(args.image_folder, x["image_path"][0]),
                            "nframes": args.max_frames,
                            "grid_size": x["grid_size"] if "grid_size" in x else None,
                            "max_pixels": args.max_pixels
                        },
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE.format(question=prompt)
                        }
                ]
            }]
        elif args.dataset in ["spar_bench", "spar_bench_tiny", "mmsi_bench"]:
            prompt = x["question"]
            if x['problem_type'] == 'multiple choice' and x.get('options', None) is not None:
                prompt = prompt + '\n' + '\n'.join(x['options'])
            prompt = prompt.replace("Your answer can only include one of options A, B, C or D.", "")
            prompt = prompt.replace("Answer using a single number and nothing else.", "")
            
            post_prompt = ""
            if x.get('original_question_type', None) in ['position_matching', "camera_motion_infer"]:
                post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
            prompt = prompt + "\n" + post_prompt 

            if x['data_type'] == 'single_view':
                prompt =  prompt + '\nThe index of the given image is 1.' + TYPE_TEMPLATE[x['problem_type'].lower()]
                width, height = fetch_image({"image": os.path.join(args.image_folder, x["image_path"][0]), "max_pixels": args.max_pixels}).size
                msg = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(width=width, height=height)
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": os.path.join(args.image_folder, x["image_path"][0]),
                                "max_pixels": args.max_pixels
                            },
                            {
                                "type": "text",
                                "text": PROMPT_TEMPLATE.format(question=prompt) 
                            }
                        ]
                    }
                ]
            elif x['data_type'] == 'multi_view':  # multi_view
                n_frames = len(x["image_path"])  
                width, height = fetch_image({"image": os.path.join(args.image_folder, x["image_path"][0]), "max_pixels": args.max_pixels}).size
                image_messages = []
                for image_idx, image_path in enumerate(x["image_path"]):
                    image_messages.extend([
                        {
                            "type": "image",
                            "image": os.path.join(args.image_folder, image_path),
                            "max_pixels": args.max_pixels
                        },
                        {
                            "type": "text",
                            "text": f"The index of the given image is {image_idx+1} (width: {width}, height: {height}).\n"
                        }
                    ])
                prompt = prompt + TYPE_TEMPLATE[x['problem_type'].lower()]
                image_messages.append({
                    "type": "text",
                    "text": PROMPT_TEMPLATE.format(question=prompt)
                })
                
                msg = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(width=width, height=height)
                    },
                    {
                        "role": "user",
                        "content": image_messages
                    }
                ]
        else:
            raise Exception(f"UNKNON args.dataset: {args.dataset}")
        messages.append(msg)
    if args.all > 1:
        # 分split
        output_dir = os.path.join(args.output_dir, f"{args.split}_{args.all}")     # args.output_dir 
    else:
        output_dir = args.output_dir 
    save_dir = output_dir
    if args.over_write:
        os.system(f"rm -rf {output_dir} && mkdir {output_dir}")
    else:
        if not os.path.exists(output_dir):
            os.system(f"mkdir {output_dir}")
    start_idx = 0
    output_file_path = f"{output_dir}/results.jsonl"
    if os.path.exists(output_file_path):
        mode = "a"
        with open(output_file_path) as fin:
            for line in fin:
               start_idx += 1
    else:
        mode = "w"
    print("Output Dir: ", output_dir)

    with open(output_file_path, mode, encoding="utf-8") as fout:
        print("Message Example:", messages[0])
        print(f"Start from the {start_idx} example")
        for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
            batch_messages = messages[i:i + BSZ]
            prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_num = []
            for msg in batch_messages:
                current_image_num = 0
                for turn in msg:
                    if isinstance(turn["content"], list):
                        for turn_content in turn["content"]:
                            if turn_content["type"] == "image":
                                current_image_num += 1
                if args.dataset in ["vsi_bench"]:
                    assert current_image_num == args.max_frames, f"wrong image number: {current_image_num} != {args.max_frames}"
                elif args.dataset in ["maze", "SpatialEval_spatialreal"]:
                    assert current_image_num == 1, f"wrong image number: {current_image_num}"
                image_num.append(current_image_num)
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            image_idx = 0
            video_idx = 0
            llm_inputs = []
            for idx, (prompt, msg) in enumerate(zip(prompts, batch_messages)):
                mm_type = batch_messages[idx][1]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = []
                    for current_idx in range(image_num[idx]):
                        width, height = image_inputs[image_idx].size
                        if args.dataset in ["video", "vsi_bench"]:   
                            sample_mm_data["image"].append(image_inputs[image_idx])             # resize(, Image.Resampling.LANCZOS)
                        else:
                            sample_mm_data["image"].append(image_inputs[image_idx])
                        image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = [video_inputs[video_idx]]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                llm_inputs.append({
                    "prompt": prompt,
                    "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                    "grid_size": msg[1]["content"][0]["grid_size"] if args.dataset == 'maze' else None
                })
            if image_inputs is not None:
                assert image_idx == len(image_inputs), f"Image index mismatch: {image_idx} != {len(image_inputs)}"
            if video_inputs is not None:
                assert video_idx == len(video_inputs), f"Video index mismatch: {video_idx} != {len(video_inputs)}"

            if i < 1e9:
                batch_sequences = multi_turn_generate(llm, tokenizer, vllm_inputs=llm_inputs, sampling_params=sampling_params, save_dir=save_dir,
                            max_num_steps=20 if args.dataset=="maze" else 10)
                batch_sequences, all_sample_dir = batch_sequences
            else:
                batch_sequences = multi_turn_generate(llm, tokenizer, vllm_inputs=llm_inputs, sampling_params=sampling_params, save_dir=None,
                            max_num_steps=20 if args.dataset=="maze" else 10)
                all_sample_dir = [None] * len(batch_sequences)
            batch_conversations = [parse_dialog(sequence) for sequence in batch_sequences]
            print(f"Processed batch {(i)//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}. ")

            for input_example, model_output, sample_dir in zip(data[i:i + BSZ], batch_conversations, all_sample_dir):
                result = input_example.copy()
                result['conversations'] = model_output
                result['model_output'] = model_output[-1]['content']
                result['model_id'] = model_name
                result['sample_dir'] = sample_dir

                fout.write(
                    json.dumps(result)
                    + "\n"
                )
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the question file")
    parser.add_argument("--output-dir", type=str, default="./result")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=256*28*28)
    parser.add_argument("--over_write", type=int, default=0, help="Whether to overwrite the output directory")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--all", type=int, default=1)
    args = parser.parse_args()
    if args.image_folder == "None":
        args.image_folder = ""
    eval_model(args)
    