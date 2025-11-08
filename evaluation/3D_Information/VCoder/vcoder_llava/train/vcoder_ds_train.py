# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional, Sequence
import numpy as np
import random
import torch
import transformers
import json

from vcoder_llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SEG_TOKEN, DEFAULT_DEPTH_TOKEN

from torch.utils.data import Dataset
from vcoder_llava.train.vcoder_ds_llava_trainer import VCoderDSLLaVATrainer

from vcoder_llava import vcoder_conversation as conversation_lib
from vcoder_llava.model import *
from vcoder_llava.mm_utils import tokenizer_image_token, tokenizer_seg_token, tokenizer_depth_seg_token
from vcoder_llava.data_utils import generate_qa_pairs
from .train import (
    get_peft_state_maybe_zero_3, 
    get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3,
    find_all_linear_names,
)
from vcoder_llava.questions import DEPTH_QUESTIONS, SEMANTIC_QUESTIONS, INSTANCE_QUESTIONS, PANOPTIC_QUESTIONS

from PIL import Image

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_projector_type: Optional[str] = field(default='linear')
    freeze_llm: bool = field(default=False)
    
    use_mm2_proj: bool = field(default=False)
    pretrain_mm2_mlp_adapter: Optional[str] = field(default=None)

    seg_tune_adapter: bool = field(default=False)
    mm_seg_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    seg_mm_projector_type: Optional[str] = field(default='linear')

    depth_tune_adapter: bool = field(default=False)
    mm_depth_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    depth_mm_projector_type: Optional[str] = field(default='linear')

    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_seg_select_feature: Optional[str] = field(default="patch")
    mm_depth_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    depth_data_path: str = field(default=None,
                           metadata={"help": "Path to the seg training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    seg_image_folder: Optional[str] = field(default=None)
    depth_image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_seg_mm_mlp_adapter: bool = field(default=False)
    freeze_depth_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def depth_seg_preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_seg: bool = False,
    has_depth: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image and has_seg:
        if has_depth:
            input_ids = torch.stack([tokenizer_depth_seg_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = torch.stack([tokenizer_seg_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and has_seg:
                if has_depth:
                    round_len = len(tokenizer_depth_seg_token(rou, tokenizer))
                    instruction_len = len(tokenizer_depth_seg_token(parts[0], tokenizer)) - 3
                else:
                    round_len = len(tokenizer_seg_token(rou, tokenizer))
                    instruction_len = len(tokenizer_seg_token(parts[0], tokenizer)) - 2
            elif has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def vcoder_ds_preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
            if DEFAULT_SEG_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SEG_TOKEN, '').strip()
                sentence['value'] = DEFAULT_SEG_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            
            replace_token = DEFAULT_SEG_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_SEG_TOKEN, replace_token)
            
            if DEFAULT_DEPTH_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_DEPTH_TOKEN, '').strip()
                sentence['value'] = DEFAULT_DEPTH_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            
            replace_token = DEFAULT_DEPTH_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_DEPTH_TOKEN, replace_token)
                    
    return sources

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_seg: bool = False,
    has_depth: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("v1"):
        return depth_seg_preprocess_v1(sources, tokenizer, has_image=has_image, has_seg=has_seg, has_depth=has_depth)
    raise ValueError(f"Unknown conversation version: {conversation_lib.default_conversation.version}")

def _obtain_depth_texts(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    
    depth_labels = {}
    for line in lines:
        key = line.split("<IMG>")[1].strip("\n")
        label = line.split("<IMG>")[2].strip("\n")
        depth_labels[key] = label
    
    return depth_labels

def _obtain_seg_texts(file_path):
    def _remove_specific_word(text, word_to_remove):
        import re
        tokens = re.findall(r'\b\w+\b|[,.]', text)
        result_tokens = []
        word_found = False

        for i, token in enumerate(tokens):
            if token == word_to_remove:
                if not word_found:
                    # Keep the first occurrence and mark it as found
                    result_tokens.append(token)
                    word_found = True
                else:
                    # Remove any preceding punctuation if it's just before this word
                    if i > 0 and tokens[i-1] in {',', '.'}:
                        result_tokens.pop()
            else:
                result_tokens.append(token)

        # Join tokens and clean up spaces before punctuation
        result_text = ' '.join(result_tokens)
        result_text = re.sub(r'\s([,.](?:\s|$))', r'\1', result_text)
        return result_text

    with open(file_path) as f:
        lines = f.readlines()
    
    seg_labels = {}
    for line in lines:
        key = line.split("<IMG>")[1].strip("\n")
        label = line.split("<IMG>")[2].strip("\n")
        label = _remove_specific_word(label, "wall")
        label = _remove_specific_word(label, "window")
        seg_labels[key] = label
    
    return seg_labels

def obtain_seg_data_splits(data_args):
    def _get_labels(folder):
        return _obtain_seg_texts(os.path.join(data_args.seg_image_folder, folder, "panoptic.txt"))
    
    list_data_dict = []
    data_dict = json.load(open(data_args.data_path, "r"))
    
    for l in data_dict:
        if "image" in l.keys():
            if os.path.exists(os.path.join(data_args.image_folder, l["image"])):
                l["seg"] = l["image"].split("/")[-1]
                if "coco" in l["image"]:
                    l["seg_folder"] = "coco_segm_text/train/panoptic_inference"
                elif "gqa" in l["image"]:
                    l["seg_folder"] = "gqa/seg_images/panoptic_inference"
                elif "VG_100K_2" in l["image"]:
                    l["seg_folder"] = "vg/vg/SEG_VG_100K_2/panoptic_inference"
                elif "VG_100K" in l["image"]:
                    l["seg_folder"] = "vg/vg/SEG_VG_100K/panoptic_inference"
                elif "ocr_vqa" in l["image"]:
                    l["seg_folder"] = "ocr_vqa/seg_images/panoptic_inference"
                if "textvqa" in l["image"]:
                    l["seg_folder"] = "textvqa/seg_images/panoptic_inference"
                conversations = []
                for c in l["conversations"]:
                    if "<image>" in c["value"]:
                        c["value"] = c["value"].replace("<image>", "<image>\n<seg>")
                    conversations.append(c)
                l["conversations"] = conversations
                if len(conversations) > 0:
                    list_data_dict.append(l)

    labels_dict = {
        "coco_segm_text/train": _get_labels("coco_segm_text/train/"),
        "gqa/seg_images": _get_labels("gqa/seg_images/"),
        "vg/vg/SEG_VG_100K": _get_labels("vg/vg/SEG_VG_100K/"),
        "vg/vg/SEG_VG_100K_2": _get_labels("vg/vg/SEG_VG_100K_2/"),
        "ocr_vqa/seg_images": _get_labels("ocr_vqa/seg_images"),
        "textvqa/seg_images": _get_labels("textvqa/seg_images/"),
    }
    
    random.shuffle(list_data_dict)
    list_data_dict = list_data_dict[:200000]
    final_list_data_dict = []
    for l in list_data_dict:
        prob_add = np.random.uniform(0,1.)
        if prob_add > 0.7:
            labels = labels_dict[l["seg_folder"].split("/panoptic_inference")[0]]
            conversations = l["conversations"]
            even_indices = list(range(2, len(conversations) + 1, 2))
            random_even_index = random.choice(even_indices)
            question_prob = np.random.uniform(0,1.)
            if question_prob > 0.90:
                question = "What objects can be seen in the image?"
            else:
                question = random.choice(PANOPTIC_QUESTIONS)
            conv = [{
                    "from": "human", 
                    "value": question
                }, 
                {
                    "from": "gpt",
                    "value": labels[l["seg"]]
                }]
            final_conversations = conversations[:random_even_index] + conv + conversations[random_even_index:]
            l["conversations"] = final_conversations
        final_list_data_dict.append(l)
    return final_list_data_dict

def obtain_seg_depth_data_splits(data_args):
    data_dict = json.load(open(data_args.data_path, "r"))
    list_data_dict = []
    labels = _obtain_depth_texts(os.path.join(data_args.depth_data_path, "coco_segm_text", "depth", "train", "panoptic_order.txt"))
    for l in data_dict:
        if "image" in l.keys():
            if os.path.exists(os.path.join(data_args.image_folder, l["image"])):
                if "coco" in l["image"]:
                    l["depth"] = l["image"].split("/")[-1]
                    l["seg"] = l["image"].split("/")[-1]
                    l["seg_folder"] = "coco_segm_text/train/panoptic_inference"
                    l["depth_folder"] = "coco_segm_text/depth/train/depth"
                    conversations = []
                    for c in l["conversations"]:
                        if "<image>" in c["value"]:
                            c["value"] = c["value"].replace("<image>", "<image>\n<seg>\n<depth>")
                        conversations.append(c)
                    l["conversations"] = conversations
                    if len(conversations) > 0:
                        list_data_dict.append(l)
    random.shuffle(list_data_dict)
    list_data_dict = list_data_dict[:100000]
    final_list_data_dict = []
    for l in list_data_dict:
        prob_add = np.random.uniform(0,1.)
        if prob_add > 0.7:
            conversations = l["conversations"]
            even_indices = list(range(2, len(conversations) + 1, 2))
            random_even_index = random.choice(even_indices)
            conv = [{
                    "from": "human", 
                    "value": random.choice(DEPTH_QUESTIONS)
                }, 
                {
                    "from": "gpt",
                    "value": labels[l["seg"]]
                }]
            final_conversations = conversations[:random_even_index] + conv + conversations[random_even_index:]
            l["conversations"] = final_conversations
        final_list_data_dict.append(l)
    return final_list_data_dict

def get_object_data_depth_split(data_args):
    list_data_dict = []
    for bucket in ["train", "unlabeled", "test"]:
            panoptic_labels = _obtain_seg_texts(os.path.join(data_args.seg_image_folder, "coco_segm_text", bucket, "panoptic.txt"))

            for key in panoptic_labels.keys():
                question_prob = np.random.uniform(0,1.)
                answer = panoptic_labels[key]
                if question_prob > 0.90:
                    question = "What objects can be seen in the image?"
                else:
                    question = random.choice(PANOPTIC_QUESTIONS)
                seg_folder = "panoptic_inference"

                question += "\n<image>\n<seg>\n<depth>"
                conversations = [ 
                        {
                            "from": "human", 
                            "value": question
                        }, 
                        {
                            "from": "gpt",
                            "value": answer
                        },
                    ]
                list_data_dict.append(
                        {
                            "conversations": conversations,
                            "image": "coco/" + bucket + "2017/" + key,
                            "seg": key,
                            "depth": key,
                            "seg_folder": "coco_segm_text/" + bucket + "/" + seg_folder,
                            "depth_folder": "coco_segm_text/depth/" + bucket + "/" + "depth"
                        }
                    )
                    
    random.shuffle(list_data_dict)
    return list_data_dict[:50000]

def get_object_data_split(data_args):
    list_data_dict = []
    for bucket in ["train", "unlabeled", "test"]:
            panoptic_labels = _obtain_seg_texts(os.path.join(data_args.seg_image_folder, "coco_segm_text", bucket, "panoptic.txt"))
            semantic_labels = _obtain_seg_texts(os.path.join(data_args.seg_image_folder, "coco_segm_text", bucket, "semantic.txt"))
            instance_labels = _obtain_seg_texts(os.path.join(data_args.seg_image_folder, "coco_segm_text", bucket, "instance.txt"))

            for key in panoptic_labels.keys():
                assert key in semantic_labels.keys() and key in instance_labels.keys(), "Instance, semantic, and panoptic labels should have the same keys."
                prob_task = np.random.uniform(0,1.)
                question_prob = np.random.uniform(0,1.)
                if prob_task < 0.33:
                    answer = semantic_labels[key]
                    if question_prob > 0.90:
                        question = "What objects can be seen in the image?"
                    else:
                        question = random.choice(SEMANTIC_QUESTIONS)
                    seg_folder = "semantic_inference"
                elif prob_task < 0.66:
                    answer = instance_labels[key]
                    if question_prob > 0.90:
                        question = "What objects can be seen in the image?"
                    else:
                        question = random.choice(INSTANCE_QUESTIONS)
                    seg_folder = "instance_inference"
                else:
                    answer = panoptic_labels[key]
                    if question_prob > 0.90:
                        question = "What objects can be seen in the image?"
                    else:
                        question = random.choice(PANOPTIC_QUESTIONS)
                    seg_folder = "panoptic_inference"

                question += "\n<image>\n<seg>"
                conversations = [ 
                        {
                            "from": "human", 
                            "value": question
                        }, 
                        {
                            "from": "gpt",
                            "value": answer
                        },
                    ]
                list_data_dict.append(
                        {
                            "conversations": conversations,
                            "image": "coco/" + bucket + "2017/" + key,
                            "seg": key,
                            "seg_folder": "coco_segm_text/" + bucket + "/" + seg_folder
                        }
                    )
                    
    random.shuffle(list_data_dict)
    return list_data_dict

def get_depth_data_split(data_args):
    list_data_dict = []
    for bucket in ["train", "unlabeled", "test"]:
            labels = _obtain_depth_texts(os.path.join(data_args.depth_data_path, "coco_segm_text", "depth", bucket, "panoptic_order.txt"))
            for key in labels.keys():
                answer = labels[key]
                question = random.choice(DEPTH_QUESTIONS)
                question += "\n<image>\n<seg>\n<depth>"
                seg_folder = "panoptic_inference"

                conversations = [ 
                        {
                            "from": "human", 
                            "value": question
                        }, 
                        {
                            "from": "gpt",
                            "value": answer
                        },
                    ]

                list_data_dict.append(
                        {
                            "conversations": conversations,
                            "image": "coco/" + bucket + "2017/" + key,
                            "seg": key,
                            "depth": key,
                            "seg_folder": "coco_segm_text/" + bucket + "/" + seg_folder,
                            "depth_folder": "coco_segm_text/depth/" + bucket + "/" + "depth"
                        }
                    )
    random.shuffle(list_data_dict)
    return list_data_dict

def get_extra_count_data_split(data_args):
    list_data_dict = []
    bucket = "train"
    panoptic_labels = _obtain_seg_texts(os.path.join(data_args.seg_image_folder, "coco_segm_text", bucket, "panoptic.txt"))

    for key in panoptic_labels.keys():
        prob = np.random.uniform(0,1.)
        if prob > 0.99:
            answer = panoptic_labels[key]
            seg_folder = "panoptic_inference"

            qa_pairs = generate_qa_pairs(answer)
            if len(qa_pairs) >= 1:
                conversations = []
                for idx, qa_pair in enumerate(qa_pairs):
                    conversations.append(
                        {
                            "from": "human",
                            "value": qa_pair[0] + "\n<image>\n<seg>" if idx == 0 else qa_pair[0]
                        }
                    )
                    conversations.append(
                        {
                            "from": "gpt",
                            "value": qa_pair[1]
                        }
                    )
                list_data_dict.append(
                        {
                            "conversations": conversations,
                            "image": "coco/" + bucket + "2017/" + key,
                            "seg": key,
                            "seg_folder": "coco_segm_text/" + bucket + "/" + seg_folder
                        }
                    )
                    
    random.shuffle(list_data_dict)
    return list_data_dict

class LazyDepthSegSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazyDepthSegSupervisedDataset, self).__init__()
        
        list_data_dict = []
        if data_args.data_path is not None:
            
            print("Preparing dataset, this may take upto 5 minutes...")
            
            seg_data_list = obtain_seg_data_splits(data_args)
            list_data_dict.extend(seg_data_list)
            
            depth_data_list = obtain_seg_depth_data_splits(data_args)
            list_data_dict.extend(depth_data_list)

            depth_object_list = get_object_data_depth_split(data_args)
            list_data_dict.extend(depth_object_list)

            object_data_list = get_object_data_split(data_args)
            list_data_dict.extend(object_data_list)

            depth_order_list = get_depth_data_split(data_args)
            list_data_dict.extend(depth_order_list)

            extra_object_list = get_extra_count_data_split(data_args)
            list_data_dict.extend(extra_object_list)
            
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer

        random.shuffle(list_data_dict)

        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            seg_tokens = 128 if 'seg' in sample else 0
            img_tokens = 128 if 'image' in sample else 0
            depth_tokens = 128 if 'depth' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens + seg_tokens + depth_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = cur_len if 'seg' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            
            seg_file = self.list_data_dict[i]['seg']
            seg_folder = self.data_args.seg_image_folder
            seg = Image.open(os.path.join(seg_folder, self.list_data_dict[i]['seg_folder'], seg_file)).convert('RGB')
            seg_processor = self.data_args.seg_image_processor

            if 'depth' in sources[0]:
                depth_file = self.list_data_dict[i]['depth']
                depth_folder = self.data_args.depth_data_path
                depth = Image.open(os.path.join(depth_folder, self.list_data_dict[i]['depth_folder'], depth_file)).convert('RGB')
                depth_processor = self.data_args.depth_image_processor
            else:
                depth = None
            
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                seg = expand2square(seg, tuple(int(x*255) for x in seg_processor.image_mean))
                seg = seg_processor.preprocess(seg, return_tensors='pt')['pixel_values'][0]
                if depth is not None:
                    depth = expand2square(depth, tuple(int(x*255) for x in depth_processor.image_mean))
                    depth = depth_processor.preprocess(depth, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                seg = seg_processor.preprocess(seg, return_tensors='pt')['pixel_values'][0]
                if depth is not None:
                    depth = depth_processor.preprocess(depth, return_tensors='pt')['pixel_values'][0]
            sources = vcoder_ds_preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            has_seg=('seg' in self.list_data_dict[i]),
            has_depth=('depth' in self.list_data_dict[i])
        )
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if 'seg' in self.list_data_dict[i]:
            data_dict['seg'] = seg
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.seg_image_processor.crop_size
            data_dict['seg'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if 'depth' in self.list_data_dict[i]:
            data_dict['depth'] = depth
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.depth_image_processor.crop_size
            data_dict['depth'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForDepthSegSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'seg' in instances[0]:
            segs = [instance['seg'] for instance in instances]
            if all(x is not None and x.shape == segs[0].shape for x in segs):
                batch['segs'] = torch.stack(segs)
            else:
                batch['segs'] = segs
        
        if 'depth' in instances[0]:
            depths = [instance['depth'] for instance in instances]
            if all(x is not None and x.shape == depths[0].shape for x in depths):
                batch['depths'] = torch.stack(depths)
            else:
                batch['depths'] = depths

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazyDepthSegSupervisedDataset(tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForDepthSegSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def vcoder_ds_train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.depth_tune_adapter is not None:
        if 'mpt' in model_args.model_name_or_path:
            raise ValueError("MPT is not supported for VCoder Adapted Training.")
        else:
            model = VCoderDSLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        raise ValueError("MPT is not supported for VCoder Adapted Training.")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # vision modules
    model.get_vision_tower().load_model()
    data_args.image_processor = model.get_vision_tower().image_processor
    data_args.is_multimodal = True

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    if model_args.seg_tune_adapter is not None:
        model.get_model().initialize_seg_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        if model_args.seg_tune_adapter:
            if getattr(model_args, "freeze_llm", False):
                model.requires_grad_(False)
            for p in model.get_model().seg_mm_projector.parameters():
                p.requires_grad = True
            for p in model.get_model().vcoder_lm_emb.parameters():
                p.requires_grad = True

        data_args.seg_image_processor = model.get_vision_tower().image_processor
        model.config.use_mm2_proj = model_args.use_mm2_proj
        model.config.mm_vcoder_lm_emb = True

        model.config.seg_tune_adapter = training_args.seg_tune_adapter = model_args.seg_tune_adapter

        model.config.freeze_seg_mm_mlp_adapter = training_args.freeze_seg_mm_mlp_adapter
        if training_args.freeze_seg_mm_mlp_adapter:
            for p in model.get_model().seg_mm_projector.parameters():
                p.requires_grad = False
        
        if model_args.use_mm2_proj:
            for p in model.get_model().mm2_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().seg_mm_projector.to(dtype=compute_dtype, device=training_args.device)
    else:
        # seg modules
        data_args.seg_image_processor = model.get_vision_tower().image_processor

        if training_args.bits in [4, 8]:
            model.get_model().seg_mm_projector.to(dtype=compute_dtype, device=training_args.device)
    
    if model_args.depth_tune_adapter is not None:
        model.get_model().initialize_depth_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        if getattr(model_args, "freeze_llm", False):
            model.requires_grad_(False)
        for p in model.get_model().depth_mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().vcoder_lm_emb.parameters():
            p.requires_grad = True

        if model_args.seg_tune_adapter:
            for p in model.get_model().seg_mm_projector.parameters():
                p.requires_grad = True
            
        data_args.depth_image_processor = model.get_vision_tower().image_processor
        model.config.depth_tune_adapter = training_args.depth_tune_adapter = model_args.depth_tune_adapter

        model.config.freeze_depth_mm_mlp_adapter = training_args.freeze_depth_mm_mlp_adapter
        if training_args.freeze_depth_mm_mlp_adapter:
            for p in model.get_model().depth_mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().depth_mm_projector.to(dtype=compute_dtype, device=training_args.device)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = VCoderDSLLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # import torch.distributed as dist
    # if dist.get_rank() == 0:
    #     from icecream import ic
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             ic(name, param.shape)
    # exit()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    vcoder_ds_train()
