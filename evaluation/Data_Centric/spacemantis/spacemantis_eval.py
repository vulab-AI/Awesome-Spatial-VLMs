import os
import json
import warnings
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, AutoModelForCausalLM 
from datasets import load_dataset

import torch

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')
from models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava


def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument("--output_path", type=str, default="/home/yxy1421/tuo/spatial_eval/output/spacemantis")
    return parser.parse_args()


def process_bench(bench_name: str, dataset, model, processor, args):
    bench_split = dataset[bench_name]
    all_outputs = []

    os.makedirs(args.output_path, exist_ok=True)
    result_path = os.path.join(args.output_path, f"{bench_name}_results.json")
    if os.path.exists(result_path):
        print(f"[{bench_name}] Results already exist, skipping...")
        return

    pbar = tqdm(total=len(bench_split), desc=f"Processing {bench_name}")
    for item in bench_split:
        image_list = []
        for key in item.keys():
            if key.startswith("image_") and item[key] is not None:
                image = item[key].convert("RGB") if isinstance(item[key], Image.Image) else Image.open(item[key]).convert("RGB")
                image = resize_max_500(image)
                image_list.append(image)
        if len(image_list) == 0:
            print("No images found in item")

        # build prompt
        prompt = item["prompt"]
        prompt += " If it's selection question, only return the option letter. Else, only return the answer phrase."

        with torch.no_grad():
            response, history = chat_mllava(
                prompt, image_list, model, processor,
                max_new_tokens=512, num_beams=1, do_sample=False, use_cache=True
            )
        output_text = response
        print("Output:", output_text)

        all_outputs.append({
            "bench_name": bench_name,
            "id": item["id"],
            "prompt": item["prompt"],
            "options": item["options"],
            "GT": item["GT"],
            "output": output_text
        })

        pbar.update(1)
        torch.cuda.empty_cache()

    pbar.close()
    with open(result_path, "w") as f:
        json.dump(all_outputs, f, indent=4)

    print(f"[{bench_name}] Done -> {result_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # load dataset
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

    # load model & processor ONCE
    attn_implementation = "flash_attention_2"
    processor = MLlavaProcessor.from_pretrained("remyxai/SpaceMantis")
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaForConditionalGeneration.from_pretrained(
        "remyxai/SpaceMantis",
        device_map="cuda",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        attn_implementation=attn_implementation
    ).eval()

    for bench_name in dataset.keys():
        process_bench(bench_name, dataset, model, processor, args)


if __name__ == "__main__":
    main()
