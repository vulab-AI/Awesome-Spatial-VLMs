

import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from datasets import load_dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# cambrian-8b
conv_mode = "llama_3" 

#load model

import torch
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = os.path.expanduser("nyu-visionx/cambrian-8b")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

temperature = 0


def resize_max_500(image: Image.Image,max_size=280) -> Image.Image:
    """resize to 280 max size"""
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    if width >= height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height), Image.LANCZOS)

def process(images, question, tokenizer, image_processor, model_config):
    qs = question

    # insert image token for each image
    if model_config.mm_use_im_start_end:
        image_tokens = "".join(
            [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN for _ in images]
        )
    else:
        image_tokens = "".join([DEFAULT_IMAGE_TOKEN for _ in images])

    qs = image_tokens + '\n' + qs

    # build conversation
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # image process
    image_sizes = [img.size for img in images]
    image_tensors = process_images(images, image_processor, model_config)

    # tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    return input_ids, image_tensors, image_sizes, prompt


import torch
from PIL import Image

def inference(images, question, tokenizer, image_processor, model, model_config, temperature=0.2, device="cuda"):
    # support single image or list of images
    if not isinstance(images, list):
        images = [images]

    # input process
    input_ids, image_tensors, image_sizes, prompt = process(
        images, question, tokenizer, image_processor, model_config
    )

    input_ids = input_ids.to(device, non_blocking=True)

    # ====== inference ======
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output/cambrian")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load dataset
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

    image_list = [f"image_{i}" for i in range(32)]
    for bench in dataset.keys():
        all_outputs = []
        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)
        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue

        for item in dataset[bench]:
            # get images
            images = []
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)
                    images.append(image)

            if len(images) == 0:
                continue
            
            if len(images) >1:
                continue
            # get prompt
            prompt = item.get("prompt", "")
            prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. Answer:"

            print("Prompt:", prompt)
            answer = inference(images, prompt, tokenizer, image_processor, model, model.config, temperature=0)


            print("Output:", answer, flush=True)
            print("\n", flush=True)

            all_outputs.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": item["prompt"],
                "options": item["options"],
                "GT": item["GT"],
                "result": answer
            })

            torch.cuda.empty_cache()

        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)
