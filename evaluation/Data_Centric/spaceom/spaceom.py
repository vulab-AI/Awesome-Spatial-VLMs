from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq, LlavaForConditionalGeneration,LlavaNextProcessor, LlavaNextForConditionalGeneration,LlavaOnevisionForConditionalGeneration
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO

# Configuration




# # Load and preprocess image
# if image_path.startswith("http"):
#     image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
# else:
#     image = Image.open(image_path).convert("RGB")
# if image.width > 512:
#     ratio = image.height / image.width
#     image = image.resize((512, int(512 * ratio)), Image.Resampling.LANCZOS)

# # Format input
# chat = [
#     {"role": "system", "content": [{"type": "text", "text": system_message}]},
#     {"role": "user", "content": [{"type": "image", "image": image},
#                                 {"type": "text", "text": prompt}]}
# ]
# text_input = processor.apply_chat_template(chat, tokenize=False,
#                                                   add_generation_prompt=True)

# # Tokenize
# inputs = processor(text=[text_input], images=[image],
#                                       return_tensors="pt").to("cuda")

# # Generate response
# generated_ids = model.generate(**inputs, max_new_tokens=1024)
# output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



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


def make_message_prompt(item):
    image_list = [f"image_{i}" for i in range(32)]
    
    # 构造 system message（可选）
    message = [
        {
            "role": "system", 
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
                        "You should first think about the reasoning process and then provide the answer. "
                        "Use <think>...</think> and <answer>...</answer> tags."
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
        "content": user_content  # ✅ 一定要加 content
    })

    return message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str,default="output/spaceom")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()



#make it batchsize = 50
def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    #load dataset
    dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    #load  model
    model_id = "remyxai/SpaceOm"

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map="cuda:1", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id)

    for bench in dataset.keys():
        messages = []
        id_list = []

        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)

        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue


        for item in dataset[bench]:
            message = make_message_prompt(item)
            messages.append(message)
            id_list.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": item["prompt"],
                "options": item["options"],
                "answer": item["GT"]
            })

        print("processed bench:", bench, "with", len(messages), "messages.")
        all_outputs = []

        batch_size = args.batch_size
        i = 0
        pbar = tqdm(total=len(messages), desc=f"Processing {bench}")

        while i < len(messages):
            current_batch_size = min(batch_size, len(messages) - i)
            success = False

            while not success:
                batch_messages = messages[i:i + current_batch_size]
                batch_id_list = id_list[i:i + current_batch_size]

                text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
                image_inputs, _ = process_vision_info(batch_messages)
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                model.eval()
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    batch_output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    print("batch_output_text", batch_output_text, flush=True)

                    for output_text, item_id in zip(batch_output_text, batch_id_list):
                        output_text = output_text.strip()
                        all_outputs.append({
                            "bench_name": item_id["bench_name"],
                            "id": item_id["id"],
                            "prompt": item_id["prompt"],
                            "options": item_id["options"],
                            "GT": item_id["answer"],
                            "result": output_text
                        })

                    success = True
                    i += current_batch_size
                    pbar.update(current_batch_size)

                    # 如果之前缩小过batch size，恢复到原始大小
                    if batch_size < args.batch_size:
                        batch_size = args.batch_size

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM with batch size {current_batch_size}, reducing batch size by half and retrying...")
                        torch.cuda.empty_cache()
                        current_batch_size = current_batch_size // 2
                        if current_batch_size == 0:
                            raise RuntimeError("Batch size reduced to 0, cannot proceed.")
                    else:
                        raise e

            torch.cuda.empty_cache()

        pbar.close()

        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    main()