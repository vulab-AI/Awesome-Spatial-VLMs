from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
from datasets import load_dataset
import torch
from PIL import Image
import requests
from io import BytesIO
import tempfile



def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
    width, height = image.size
    # If both dimensions are already within the limit, return the original image
    if max(width, height) <= max_size:
        return image

    if width >= height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def make_message_prompt(item):
    image_list = [f"image_{i}" for i in range(32)]
    
    # build system message
    message = [
        {
            "role": "system", 
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Answer the question with only the choice letter in the format <answer></answer>. "
                    )
                }
            ]
        }
    ]
    
    # build user message
    user_content = []

    for image_key in image_list:
        image_obj = item.get(image_key, None)
        if image_obj is not None:
            image = image_obj.convert("RGB")
            image = resize_max_500(image)
            user_content.append({
                "type": "image",
                "image": image
            })

    prompt = item.get("prompt", "")
    user_content.append({
        "type": "text",
        "text": prompt
    })

    message.append({
        "role": "user",
        "content": user_content  # ✅ must include content
    })

    return message


def save_image_and_prompt(dataset,prefix="images/"):
    for bench in dataset.keys():
        print("Processing bench:", bench,flush=True)
        bench_dataset=[]

        if f"{bench}.json" in os.listdir("."):
                print(f"{bench}.json exists,  skip")
                continue    
        
        for item in dataset[bench]:
            image_uris = []
            image_list = [f"image_{i}" for i in range(32)]
            
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)  
                    image_id = item["id"]

                    # === define local file path ===
                    if not os.path.exists(f"{prefix}{bench}"):
                        os.makedirs(f"{prefix}{bench}")
                    
                        
                    destination = f"{prefix}{bench}/{image_id}_{image_key}.png"

                    # === save image ===
                    image.save(destination, format="PNG")

                    image_uri = f"{prefix}{destination}"
                    image_uris.append(image_uri)


            bench_dataset.append({
                "bench_name":bench,
                "id": item["id"],
                "image_uris": image_uris,
                "prompt": item["prompt"]
            })
        # SAVE JSON
        json_filename = f"{bench}.json"

        with open(json_filename, 'w') as f:
            json.dump(bench_dataset, f, indent=4)
        print(f"Saved {json_filename}",flush=True)

if __name__ == "__main__":
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    save_image_and_prompt(dataset)
    





                    