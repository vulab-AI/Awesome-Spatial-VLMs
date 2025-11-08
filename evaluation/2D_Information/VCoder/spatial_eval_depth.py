# import os

# # Switch CUDA path
# os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# # Specify GPU device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use only GPU0

# import torch
# print(torch.version.cuda)   # Check CUDA version

import argparse
import os
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
import json
from datasets import load_dataset
from PIL import Image

from vcoder_llava.model.builder import load_pretrained_model

import argparse
import json
import torch
from vcoder_llava.model import *
from vcoder_llava.utils import server_error_msg
from vcoder_llava.model.builder import load_pretrained_model
from vcoder_llava.mm_utils import process_images, load_image_from_base64, tokenizer_seg_token, tokenizer_depth_seg_token, tokenizer_image_token, KeywordsStoppingCriteria
from vcoder_llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    SEG_TOKEN_INDEX, DEFAULT_SEG_TOKEN,
    DEPTH_TOKEN_INDEX, DEFAULT_DEPTH_TOKEN,
)
from transformers import TextIteratorStreamer, AutoImageProcessor, AutoModelForDepthEstimation
from threading import Thread


def resize_max(image: Image.Image, max_size: int = 384) -> Image.Image:
    """Resize the image so that the longer side is no more than max_size"""
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

def build_prompt(images, depths, prompt):
    image_token = "<image>"
    image_tokens = " ".join([image_token] * len(images))
    if depths is not None:
        depth_token = "<depth>"
        depth_tokens = " ".join([depth_token] * len(images))
        full_prompt = f"{image_tokens} {depth_tokens} {prompt}"
    else:
        full_prompt = f"{image_tokens} {prompt}"
    
    prefix = "USER: "
    suffix = "ASSISTANT: "
    full_prompt = prefix + "\n" + full_prompt + "</s>\n" + suffix
    return full_prompt


def inference(images, depths, text, tokenizer, model, image_processor, depth_processor, model_cfg):
    """
    images: list of PIL Images
    text: str, user query
    tokenizer: tokenizer for the model
    model: multimodal model
    image_processor: image preprocessing function
    model_cfg: model config (model.config)
    """
    # Build the prompt
    prompt = build_prompt(images, depths, text)
    print("Full Prompt:", prompt)

    # Tokenize text
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

    # Process images
    images = process_images(images, image_processor, model_cfg=model_cfg)
    images = images.to(dtype=torch.float16, device=model.device)

    # Process depth maps
    depths = process_images(depths, depth_processor, model_cfg=model_cfg)
    depths = depths.to(dtype=torch.float16, device=model.device)
    num_depths = depths.shape[0]

    depths_list = []
    for i in range(num_depths):
        depths_list.append(depths[i].unsqueeze(0))

    image_args = {
        "images": images,
        "segs": None,
        "depths": depths_list,
    }

    # Generate response
    generated_text = model.generate(
        inputs=input_ids,
        do_sample=False,
        max_new_tokens=1024,
        use_cache=False,
        **image_args
    )

    # Decode and extract the final answer
    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    answer = generated_text.split("ASSISTANT:")[-1].strip()

    return answer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output/vcoder_depth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load model
    tokenizer, model, image_processor, seg_processor, depth_processor, context_len = \
        load_pretrained_model("shi-labs/vcoder_ds_llava-v1.5-7b", model_base=None,                   
            model_name="vcoder_ds_llava", device_map="cuda:0")
    
    # Load depth-anything model
    depth_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf")
    depth_model = depth_model.to("cuda:0")
    
    # Load dataset
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
            # Load images
            images = []
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max(image)
                    images.append(image)

            if len(images) == 0:
                continue

            # Get prompt
            prompt = item.get("prompt", "")
            # prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. "
            # print("Input Prompt:", prompt, flush=True)

            # Generate depth maps
            depths = []
            for img in images:
                depth_input = depth_image_processor(img, return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    depth_output = depth_model(**depth_input)
                    depth = depth_output.predicted_depth
                prediction = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=images[0].size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_map = prediction.squeeze().cpu().numpy()

                # Convert to PIL Image
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                depth_map = (depth_map * 255).astype("uint8")
                depth_map = Image.fromarray(depth_map)
                depth_map = depth_map.convert("RGB")  # ← Convert to RGB

                depths.append(depth_map)

            # Run inference
            answer = inference(images, depths, prompt, tokenizer, model, image_processor, depth_processor, model.config)
            
            # Print the result
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

            # Release GPU memory
            torch.cuda.empty_cache()

        # Save results
        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)