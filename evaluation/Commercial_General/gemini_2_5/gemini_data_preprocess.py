from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
from datasets import load_dataset
import bitsandbytes 
import torch
from PIL import Image
import requests
from io import BytesIO
from google.cloud import storage
import tempfile



def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
    width, height = image.size
    # if both dimensions are within the limit, return original image
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
    
    # use system message
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
    
    # 构造 user message
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


#save PIL image to gstorage
#bucket_name: your gstorage bucket name
#prefix: the prefix path in the bucket
def save_image_to_gstorage(dataset, bucket_name="your-bucket-name", prefix="images/"):
    # init GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for bench in dataset.keys():
        print("Processing bench:", bench,flush=True)
        bench_dataset=[]
        for item in dataset[bench]:
            image_uris = []
            image_list = [f"image_{i}" for i in range(32)]
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)   
                    image_id = item["id"]

                    # ===  PIL image -> bytes ===
                    img_bytes = BytesIO()
                    image.save(img_bytes, format="PNG")
                    img_bytes = img_bytes.getvalue()

                    # === define GCS path ===
                    destination = f"{prefix}{bench}/{image_id}_{image_key}.png"
                    blob = bucket.blob(destination)

                    # === upload ===
                    blob.upload_from_string(img_bytes, content_type="image/png")

                    # === save gs:// address back to item ===
                    gcs_uri = f"gs://{bucket_name}/{destination}"
                    image_uris.append(gcs_uri)
                    print(f"✅ Uploaded {image_key} -> {gcs_uri}",flush=True)

            bench_dataset.append({
                "bench_name":bench,
                "id": item["id"],
                "image_uris": image_uris,
                "prompt": item["prompt"]
            })
        # save JSON file
        json_filename = f"{bench}.json"

        with open(json_filename, 'w') as f:
            json.dump(bench_dataset, f, indent=4)
        print(f"Saved {json_filename}",flush=True)

if __name__ == "__main__":
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    save_image_to_gstorage(dataset)
    





                    