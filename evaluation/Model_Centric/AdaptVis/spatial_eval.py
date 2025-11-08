import argparse
import os
import pandas as pd
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
import json
from datasets import load_dataset
from PIL import Image


def resize_max(image: Image.Image, max_size: int = 384) -> Image.Image:
    """将图像最长边缩放到 max_size 以内"""
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


def safe_inference(model, images, prompt, args):
    """带 OOM 恢复的安全推理：尝试 256 和 128，如果都失败就跳过"""
    resize_candidates = [256, 128]
    last_error = None

    for size in resize_candidates:
        try:
            resized_images = [resize_max(img, max_size=size) for img in images]
            result = model.single_inference(
                resized_images, prompt, args.method,
                weight=args.weight, threshold=args.threshold,
                weight1=args.weight1, weight2=args.weight2
            )
            return result
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ OOM detected at size {size}, trying smaller size...")
            last_error = e
            torch.cuda.empty_cache()
            continue

    # 如果两次都失败，直接跳过该样本
    print("❌ Skipping this sample due to repeated OOM.")
    return None


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="llava1.5", type=str,
                        choices=["llava1.5", "llava1.6"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--method", type=str, default="adapt_vis",
                        choices=["adapt_vis", "default", "scaling_vis"])
    parser.add_argument("--dola-decoding", action="store_true")
    parser.add_argument("--info-layer", type=int)
    parser.add_argument("--download", action="store_true",
                        help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true",
                        help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-path", default="output/adapt_vis", type=str)
    parser.add_argument("--weight", default=1.0, type=float)
    parser.add_argument("--weight1", default=1.0, type=float)
    parser.add_argument("--weight2", default=1.0, type=float)
    parser.add_argument("--threshold", default=1.0, type=float)
    parser.add_argument("--option", default='four', type=str,
                        choices=['two', 'four', 'six'])
    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    print(args)
    print("Loading model...")
    model, image_preprocess = get_model(args.model_name, args.device, args.method)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # 加载数据集
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
            # 获取图像
            images = []
            image_token = "<image>"
            index = 0
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max(image, 256)  # 默认先缩放到 256
                    images.append(image)
                    if index > 0:
                        image_token += f" <image> "
                    index += 1

            if len(images) == 0:
                continue

            # 构建 prompt
            prompt = "User: " + image_token + "\n"
            prompt = prompt + item.get("prompt", "")
            prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. Assistant:"

            print("Prompt:", prompt)

            # 推理（自动 OOM 降级，如果失败就跳过）
            answer = safe_inference(model, images, prompt, args)

            if answer is None:
                continue  # 跳过当前样本

            # 打印结果
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

            # 清理显存
            del images
            del answer
            torch.cuda.empty_cache()

        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    args = config()
    main(args)
