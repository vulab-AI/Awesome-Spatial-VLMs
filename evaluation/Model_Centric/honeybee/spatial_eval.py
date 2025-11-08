from PIL import Image
import matplotlib.pyplot as plt

import torch

from pipeline.interface import get_model

import json
import os
from datasets import load_dataset
import argparse


# your model path
ckpt_path = "7B-C-Abs-M256_PT/last/"

# Load model, tokenizer, and processor
model, tokenizer, processor = get_model(ckpt_path, use_bf16=True)
model.cuda()
print("Model initialization is done.")

def resize_max_500(image: Image.Image, max_size=384) -> Image.Image:
    """Resize image so that the longer side is no more than max_size (default=384)"""
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


def construct_input_prompt(user_prompt, image_list):
    """
    Construct a full prompt including system message, image placeholders, and user prompt
    """
    SYSTEM_MESSAGE = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    )

    # Insert <image> placeholder for each image
    IMAGE_TOKENS = "".join(["Human: <image>\n" for _ in image_list])

    USER_PROMPT = f"Human: {user_prompt}\n"

    return SYSTEM_MESSAGE + IMAGE_TOKENS + USER_PROMPT + "AI: "

def inference(model, tokenizer, processor, images, prompt, 
              do_sample=False, top_k=5, max_length=256, device=None):
    """
    Multi-image + prompt inference function

    Args:
        model: the pre-loaded model
        tokenizer: corresponding tokenizer
        processor: processor for image + text
        images: list of PIL.Image, input images
        prompt: str, user question
        do_sample: bool, whether to use sampling
        top_k: int, top-k sampling
        max_length: int, max generation length
        device: str, device ("cuda" or "cpu"), default is model.device
    """
    if not isinstance(images, list):
        images = [images]

    prompts = [prompt]
    inputs = processor(texts=prompts, images=images, return_tensors="pt")

    # Adjust dtype and move to device
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    if device is None:
        device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = dict(
        do_sample=do_sample,
        top_k=top_k,
        max_new_tokens=max_length
    )

    # Disable gradient computation
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)

    # Decode result to string
    sentence = tokenizer.batch_decode(res, skip_special_tokens=True)
    return sentence

def parse_args():
    # Parse output path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output/honeybee")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load dataset from HuggingFace
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

    # Define keys for all images
    image_list = [f"image_{i}" for i in range(32)]

    # Loop over each benchmark in the dataset
    for bench in dataset.keys():
        all_outputs = []
        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)

        # Skip if results already exist
        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue

        # Loop over items in the dataset
        for item in dataset[bench]:
            images = []
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)
                    images.append(image)

            if len(images) == 0:
                continue

            # Build prompt
            prompt = item.get("prompt", "")
            prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. Answer:"

            input_prompt = construct_input_prompt(prompt, images)
            print("Input Prompt:", input_prompt, flush=True)

            # Inference
            answer = inference(model, tokenizer, processor, images, input_prompt,
                              do_sample=False)

            # Print result
            print("Output:", answer, flush=True)
            print("\n", flush=True)

            # Append to result list
            all_outputs.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": item["prompt"],
                "options": item["options"],
                "GT": item["GT"],
                "result": answer
            })

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Save results to file
        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)


# -------------------------------
# The following commented block shows manual usage of processor and generation
# prompts = [prompt]
# inputs = processor(texts=prompts, images=images)
# inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
# inputs = {k: v.to(model.device) for k, v in inputs.items()}
# generate_kwargs = {
#     'do_sample': True,
#     'top_k': 5,
#     'max_length': 512
# }
# with torch.no_grad():
#     res = model.generate(**inputs, **generate_kwargs)
# sentence = tokenizer.batch_decode(res, skip_special_tokens=True)