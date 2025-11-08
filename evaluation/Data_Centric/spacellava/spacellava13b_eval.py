import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import io
import base64
import torch
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
#GCCore/13.3.0  
#Cmake/3.24.4

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


def make_message_prompt(item):
    image_list = [f"image_{i}" for i in range(32)]
    
    message = [
        {
            "role": "system", 
            "content": "You are a helpful assistant. When the user asks a question, "
                       "If it's selection question, only return the option letter. "
                       "If there is no option, only return the answer phrase."
        }
    ]
    
    user_content = []
    for image_key in image_list:
        image_obj = item.get(image_key, None)
        if isinstance(image_obj, np.ndarray):
            image_obj = Image.fromarray(image_obj)   # 🔑 dataset 可能给 numpy
        if isinstance(image_obj, Image.Image):
            image = resize_max_500(image_obj.convert("RGB"))
            user_content.append({"type": "image", "image": image})

    prompt = item.get("prompt", "")
    user_content.append({"type": "text", "text": prompt})

    message.append({"role": "user", "content": user_content})
    return message




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="/home/yxy1421/tuo/spatial_eval/output/spacellava_13b")
    return parser.parse_args()





args = parse_args()
os.makedirs(args.output_path, exist_ok=True)

# load dataset
dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

# load model & processor (gguf)
mmproj="content/mmproj-model-f16.gguf"
model_path="content/ggml-model-q4_0.gguf"
chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=True)
spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=4096, logits_all=True, n_gpu_layers = -1)



def process_bench(bench_name: str, dataset, args):
    bench_split = dataset[bench_name]
    all_outputs = []

    os.makedirs(args.output_path, exist_ok=True)
    result_path = os.path.join(args.output_path, f"{bench_name}_results.json")
    if os.path.exists(result_path):
        print(f"[{bench_name}] Results already exist, skipping...")
        return

    pbar = tqdm(total=len(bench_split), desc=f"Processing {bench_name}")
    for item in bench_split:
        
        message = make_message_prompt(item)
        results=spacellava.create_chat_completion(messages =message)
        output_text = results["choices"][0]["message"]["content"]

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

    pbar.close()
    with open(result_path, "w") as f:
        json.dump(all_outputs, f, indent=4)

    print(f"[{bench_name}] Done -> {result_path}")

# process each benchmark
for bench_name in dataset.keys():
    print(f"Processing benchmark: {bench_name}")
    process_bench(bench_name, dataset, args)











# import io
# import base64
# import numpy as np
# import torch
# from PIL import Image

# from llama_cpp import Llama
# from llama_cpp.llama_chat_format import Llava15ChatHandler


# def resize_max_500(image: Image.Image) -> Image.Image:
#     max_size = 500
#     width, height = image.size
#     # 如果最大边长已经<=1080，直接返回原图
#     if max(width, height) <= max_size:
#         return image

#     if width >= height:
#         new_width = max_size
#         new_height = int(height * max_size / width)
#     else:
#         new_height = max_size
#         new_width = int(width * max_size / height)

#     resized_image = image.resize((new_width, new_height), Image.LANCZOS)
#     return resized_image

# def image_to_base64_data_uri(image_input):
#     # Check if the input is a file path (string)
#     if isinstance(image_input, str):
#         with open(image_input, "rb") as img_file:
#             base64_data = base64.b64encode(img_file.read()).decode('utf-8')

#     # Check if the input is a PIL Image
#     elif isinstance(image_input, Image.Image):
#         buffer = io.BytesIO()
#         image_input.save(buffer, format="PNG")  # You can change the format if needed
#         base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

#     else:   
#         raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

#     return f"data:image/png;base64,{base64_data}"


# mmproj="content/mmproj-model-f16.gguf"
# model_path="content/ggml-model-q4_0.gguf"
# chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=True)
# spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=4096, logits_all=True, n_gpu_layers = -1)


# image_path='rgb.jpg'
# #resize to max 500
# image = Image.open(image_path)
# resized_image = resize_max_500(image)
# #save resized image
# resized_image.save("resized_image.png")
# image_path="resized_image.png"

# prompt="Are these 4 images the same? And describe the image."

# data_uri = image_to_base64_data_uri(image_path)
# messages = [
#     {"role": "system", "content": "You are an assistant who perfectly describes images."},
#      {
#          "role": "user",
#          "content": [
#              {"type": "image", "image": resized_image},
#              {"type": "image", "image": resized_image},
# {"type": "image", "image": resized_image},
#              {"type": "image", "image": resized_image},
#               {"type" : "text", "text": prompt}
#              ]
#          }
#     ]
# results = spacellava.create_chat_completion(messages = messages)

# #close
# print(results["choices"][0]["message"]["content"])

# #不加会有error
# spacellava.close()