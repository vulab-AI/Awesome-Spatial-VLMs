import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from PIL import Image

from ross.model.builder import load_pretrained_model
from ross.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from ross.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
from datasets import load_dataset


import requests
from io import BytesIO


def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 500
    width, height = image.size
    # resize imagesize to 500
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


def make_message(item):
    image_list = [f"image_{i}" for i in range(32)]

    new_images = []
    for image_key in image_list:
        image_obj = item.get(image_key, None)
        if image_obj is not None:
            image = image_obj.convert("RGB")
            image = resize_max_500(image)
            new_images.append(image)
    
    prompt = item.get("prompt", "")
    prompt =prompt + "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. Answer:"
    
    return prompt, new_images

    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str,default="output/ross")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()



#make it batchsize = 50
def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    #load dataset
    dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

    # Load model and processor
    model_path = "HaochenWang/ross-qwen2-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    for bench in dataset.keys():
        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)

        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue

        #each bench
        all_outputs = []
        for item in dataset[bench]:
            prompt, new_images = make_message(item)

            images_tensor = process_images(
                new_images,
                image_processor,
                model.config,
            ).cuda()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    max_new_tokens=128,
                    use_cache=True,
                    eos_token_id=151645
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("Prompt:", prompt, flush=True)
            print(outputs, flush=True)

            output_text = outputs.split("\n")[0].strip()
            all_outputs.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": prompt,
                "options": item["options"],
                "GT": item["GT"],
                "result": output_text
            })

        print("processed bench:", bench)
        all_outputs = []


        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    main()