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

# IMPORTANT: do not import torch.cuda or query devices at module import time
import torch
import multiprocessing as mp

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# -------------------------------
# Helpers (CPU-only, safe at import)
# -------------------------------
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
    parser.add_argument("--output_path", type=str, default="/home/yxy1421/tuo/spatial_eval/output/spatial_florence2")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


# -------------------------------
# Worker globals set by initializer
# -------------------------------
ARGS = None
DATASET = None


def init_worker(args):
    """
    Runs in each spawned worker exactly once.
    NOTE: Do NOT touch CUDA here.
    """
    global ARGS, DATASET
    ARGS = args
    # Load the dataset once per worker (cheap & avoids pickling large objects)
    DATASET = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")


def process_bench(bench_name: str):
    """
    Runs inside a spawned process. It's now safe to touch CUDA here.
    """
    assert torch.cuda.is_available(), "CUDA not available!"
    device = torch.device("cuda")

    # (Optional) print after CUDA context created in this process
    try:
        print(f"✅ [{bench_name}] CUDA device:", torch.cuda.get_device_name(device))
    except Exception:
        pass

    # ---- Load VLM model & tokenizer (on GPU) ----
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("remyxai/SpaceFlorence-2", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("remyxai/SpaceFlorence-2", trust_remote_code=True)

    # ---- Process dataset split ----
    global DATASET, ARGS
    bench_split = DATASET[bench_name]
    all_outputs = []

    os.makedirs(ARGS.output_path, exist_ok=True)
    result_path = os.path.join(ARGS.output_path, f"{bench_name}_results.json")
    if os.path.exists(result_path):
        print(f"[{bench_name}] Results already exist, skipping...")
        return

    pbar = tqdm(total=len(bench_split), desc=f"Processing {bench_name}")
    for item in bench_split:
        if item.get('image_1') is not None:
            #because same spatial model only support 1 or 2 images.
            pbar.update(1)
            continue

        img = item['image_0'].convert('RGB')
        image = resize_max_500(img)

        # build prompt
        prompt = item["prompt"]
        text = (
            "<SpatialVQA> Answer the question below. If it's selection question, only return the option letter. Else, only return the answer phrase."
            f"{prompt}:"
        )
        inputs = processor(text=text, images=image, return_tensors="pt").to(device, torch_dtype)


        # generate
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        output_text = processor.post_process_generation(generated_text, task="<SpatialVQA>", image_size=(image.width, image.height))

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

    # Use a spawn context and per-worker initializer (loads dataset once per worker)
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.batch_size,
        initializer=init_worker,
        initargs=(args,)
    ) as pool:
        # Get split names without pickling the dataset
        # We'll read the split inside each worker via the global DATASET
        # by name only.
        tmp_ds = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
        bench_list = list(tmp_ds.keys())
        tmp_ds = None  # free in parent

        pool.map(process_bench, bench_list)


if __name__ == "__main__":
    # CRITICAL: force spawn so CUDA is initialized only in child processes
    mp.set_start_method("spawn", force=True)
    main()
