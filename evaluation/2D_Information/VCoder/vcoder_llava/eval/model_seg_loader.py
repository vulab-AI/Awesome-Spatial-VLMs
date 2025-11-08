import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random
import glob

from vcoder_llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    SEG_TOKEN_INDEX, DEFAULT_SEG_TOKEN,
) 
from vcoder_llava.vcoder_conversation import conv_templates, SeparatorStyle
from vcoder_llava.model.builder import load_pretrained_model
from vcoder_llava.utils import disable_torch_init
from vcoder_llava.mm_utils import process_images, tokenizer_seg_token, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
from vcoder_llava.questions import QUESTIONS

import math
from PIL import Image

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, args, seg_image_folder, tokenizer, image_processor, seg_image_processor, model_config):
        self.questions = questions
        self.image_folder = args.image_folder
        self.seg_image_folder = seg_image_folder
        
        self.images = glob.glob(os.path.join(args.image_folder, '*.jpg'))
        self.images = get_chunk(self.images, args.num_chunks, args.chunk_idx)

        if seg_image_folder is not None:
            self.seg_images = glob.glob(os.path.join(seg_image_folder, '*.jpg'))
            self.seg_images = get_chunk(self.seg_images, args.num_chunks, args.chunk_idx)
            assert len(self.images) == len(self.seg_images), f"Number of images ({len(self.images)}) and seg images ({len(self.seg_images)}) must be the same"
        else:
            self.seg_images = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seg_image_processor = seg_image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        image_file = self.images[index]
        if self.seg_images is not None:
            seg_image_file = self.seg_images[index]
        else:
            seg_image_file = None
        ques = random.choice(self.questions)
        qs = DEFAULT_IMAGE_TOKEN + '\n' + ques

        image = Image.open(os.path.join(image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        if seg_image_file is not None:
            seg_image = Image.open(os.path.join(seg_image_file)).convert('RGB')
            seg_image_tensor = process_images([seg_image], self.seg_image_processor, self.model_config)[0]
            qs = DEFAULT_SEG_TOKEN + '\n' + qs
        else:
            seg_image_tensor = image_tensor
            qs = qs + " Return the answer in the paragraph format: 'The objects present in the image are: ...' and then list the objects with their count in word format (if greater than 1) in front of them, like 'two people'."
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if seg_image_file is None:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        else:
            input_ids = tokenizer_seg_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, seg_image_tensor, image_file.split("/")[-1], ques

    def __len__(self):
        return len(self.images)


# DataLoader
def create_data_loader(questions, args, seg_image_folder, tokenizer, image_processor, seg_image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, args, seg_image_folder, tokenizer, image_processor, seg_image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args, task):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, seg_image_processor, _, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = QUESTIONS[task]
    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers_file = answers_file + f'_{task}_{args.num_chunks}_{args.chunk_idx}.txt'

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    if not args.use_seg:
        seg_image_folder = None
    else:
        seg_image_folder = os.path.join(args.seg_image_folder, f'{task}_inference')

    data_loader = create_data_loader(questions, args, seg_image_folder, tokenizer, image_processor, seg_image_processor, model.config)

    for input_ids, image_tensor, seg_image_tensor, image_file, ques in tqdm(data_loader, total=len(data_loader), desc=f'Generating {task} answers...'):
        
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if "vcoder" in args.model_path:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    segs=seg_image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    depths=None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=512,
                    use_cache=True)
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=512,
                    use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        outputs = outputs.strip('\n')

        with open(f'{answers_file}', 'a') as f:
            f.write(f'Image: {image_file[0]}\n')
            f.write(f'<<QUESTION>>: {ques[0]}\n')
            f.write(f'<<ANSWER>>: {outputs}\n')
            f.write('-------------------------------------------------------\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument("--seg-image-folder", type=str, default="")
    parser.add_argument("--output-file", type=str, default="output")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    for task in ["semantic", "instance", "panoptic"]:
        eval_model(args, task)
