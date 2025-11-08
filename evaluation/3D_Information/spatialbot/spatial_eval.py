import os
import json
import warnings
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    parser.add_argument('--max_pixels', type=int, default=401408)
    parser.add_argument('--min_pixels', type=int, default=401408)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument("--output_path", type=str, default="/home/yxy1421/tuo/spatial_eval/output/spatialbot")
    parser.add_argument("--batch_size", type=int, default=6)
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
    model_name = 'RussRobin/SpatialBot-3B'
    offset_bos = 0

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.to(device).eval()

    base = model.get_model() if hasattr(model, "get_model") else model

    # Trigger any lazy init (safe on GPU in this process)
    dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
    _ = model.process_images([dummy_image, dummy_image], model.config)

    # Move submodules that might stay on CPU by default
    if hasattr(base, "mm_projector") and base.mm_projector is not None:
        base.mm_projector.to(device=device, dtype=model.dtype)

    if hasattr(base, "vision_tower") and base.vision_tower is not None:
        vt = base.vision_tower[0] if isinstance(base.vision_tower, (list, tuple)) else base.vision_tower
        if hasattr(vt, "to"):
            vt.to(device)
        if hasattr(vt, "vision_tower") and hasattr(vt.vision_tower, "to"):
            vt.vision_tower.to(device)

    for attr in ("image_encoder", "visual_encoder", "vision_model"):
        if hasattr(base, attr) and getattr(base, attr) is not None:
            enc = getattr(base, attr)
            if hasattr(enc, "to"):
                enc.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"[{bench_name}] VLM loaded")

    # ---- Load depth model on GPU ----
    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    model_zoe_nk.to(device)
    print(f"[{bench_name}] Depth model loaded")

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
            pbar.update(1)
            continue

        img = item['image_0'].convert('RGB')
        img = resize_max_500(img)

        # get depth map
        depth = model_zoe_nk.infer_pil(img)
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().cpu().numpy()
        assert isinstance(depth, np.ndarray) and depth.ndim == 2, "Depth must be 2D numpy"

        # to 16-bit png-like RGB visualization expected by your pipeline
        depth_uint16 = (depth * 256).astype(np.uint16)
        depth_img = Image.fromarray(depth_uint16)
        if len(depth_img.getbands()) == 1:
            arr = np.array(depth_img)
            h, w = arr.shape
            rgb_depth = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_depth[:, :, 0] = (arr // 1024) * 4
            rgb_depth[:, :, 1] = (arr // 32) * 8
            rgb_depth[:, :, 2] = (arr % 32) * 8
            depth_img = Image.fromarray(rgb_depth, 'RGB')

        # preprocess images
        image_tensor = model.process_images([img, depth_img], model.config)
        image_tensor = image_tensor.to(dtype=model.dtype, device=device)

        # build prompt
        prompt = item["prompt"]
        text = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "Answer the question through depth map. If it's selection question, only return the option letter. "
            "Else, only return the answer phrase. USER: <image 1>\n<image 2>\n"
            f"{prompt} ASSISTANT:"
        )
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(
            text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        # generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=100,
                use_cache=True,
                repetition_penalty=1.0
            )[0]

        output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
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
