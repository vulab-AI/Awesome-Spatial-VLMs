import datasets
from PIL import Image
import os
import numpy as np
import torch
from datasets import load_dataset


def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
    width, height = image.size
    # 如果最大边长已经<=1080，直接返回原图
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


save_dir = "./som/data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#load dataset
dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

#use the id_image_i to name the image

for bench in dataset.keys():
    id_list = []
    print("Processing bench:", bench)
    # if bench != "CV_Bench":
    #     continue
    if not os.path.exists(os.path.join(save_dir, bench)):
        os.makedirs(os.path.join(save_dir, bench))
    
    for item in dataset[bench]:
        id_list.append({
            "bench_name": bench,
            "id": item["id"],
            "prompt": item["prompt"],
            "options": item["options"],
            "answer": item["GT"]
        })
        image_list = [f"image_{i}" for i in range(32)]
        for image_key in image_list:
            image_obj = item.get(image_key, None)
            if image_obj is not None:
                image = image_obj.convert("RGB")
                image = resize_max_500(image)
                image_save_path = os.path.join(save_dir, bench, f"{item['id']}_{image_key}.jpg")
                image.save(image_save_path)
    
print("✅ All images saved to", save_dir)