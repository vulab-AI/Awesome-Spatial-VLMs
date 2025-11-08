from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq, LlavaForConditionalGeneration,LlavaNextProcessor, LlavaNextForConditionalGeneration,LlavaOnevisionForConditionalGeneration
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info, fetch_image
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# def make_question_prompt(image, question, options):

#     return (
#         f"Answer the question based on the image:\n"
#         f"Question: {question}\n"
#         f"{options}\n"
#         "Only return the answer (the option or a answer phrase)."
#     )
def resize_max_500(image: Image.Image) -> Image.Image:
    max_size = 384
    width, height = image.size
    # If the maximum side length <= max_size, return the original image
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
    
    # Construct system message (optional)
    message = [
        {
            "role": "system", 
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "You are a helpful assistant."
                    )
                }
            ]
        }
    ]
    
    # Construct user message
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
    prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase."
    user_content.append({
        "type": "text",
        "text": prompt
    })

    message.append({
        "role": "user",
        "content": user_content  # ✅ must include content
    })

    return message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str,default="output/vilasr")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()



def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    # Load dataset
    dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    # dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks_ViewSpatial-Bench", download_mode="force_redownload")
    # Load qwen2.5 model

    model_path = "inclusionAI/ViLaSR"   # The model you want to evaluate
    device_count = torch.cuda.device_count()

    # Initialize vLLM engine
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=device_count,
        limit_mm_per_prompt={"image": 10},   # Max number of images per prompt
        gpu_memory_utilization=0.85,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer


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
                llm_inputs = [{
                    "prompt": text[0],
                    "prompt_token_ids": tokenizer.encode(text[0], add_special_tokens=False),
                    "multi_modal_data": {"image": image_inputs},   # Pass images
                }]
                # ---------------------
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=512,
                )

            
                try:
                    outputs = llm.generate(prompts=llm_inputs, sampling_params=sampling_params)
                    response = outputs[0].outputs[0].text
                    print("Output:", response,flush=True)
                    print("\n",flush=True)

                    for output_text, item_id in zip([response], batch_id_list):
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

                    # If batch size was reduced due to OOM, restore it to original size
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