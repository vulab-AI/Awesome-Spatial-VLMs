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
from transformers import BitsAndBytesConfig
import os
import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

import warnings
import argparse
from modeling_bailing_qwen2_5 import Bailing_qwen2_5NativeForConditionalGeneration
from processing_bailing_qwen2_5 import Bailing_qwen2_5Processor

warnings.filterwarnings("ignore")

class BailingMMInfer:
    def __init__(self,
        model_name_or_path,
        device="cuda",
        max_pixels=None,
        min_pixels=None,
        video_max_pixels=768 * 28 * 28,
        video_min_pixels=128 * 28 * 28,
        generation_config=None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.device = device

        self.device_map = device

        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else 768 * 28 * 28
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else 128 * 28 * 28

        self.model, self.tokenizer, self.processor = self.load_model_processor()
        if max_pixels is not None:
            self.processor.max_pixels = max_pixels
        if min_pixels is not None:
            self.processor.min_pixels = min_pixels
        if generation_config is None:
            generation_config = {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 0.9
            }

        self.generation_config = generation_config


    def load_model_processor(self):
        
        model = Bailing_qwen2_5NativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            # device_map=self.device_map,
            device_map="auto",
            _attn_implementation="flash_attention_2"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_bos_token=True, trust_remote_code=True)
        processor = Bailing_qwen2_5Processor.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, video_inputs = self.processor.process_vision_info(messages)


        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        # print(inputs)
        print(self.tokenizer.decode(inputs['input_ids'][0]))

        inputs = inputs.to(self.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.generation_config,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return output_text

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
                        "You are a helpful assistant. When the user asks a question, "
                        "your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, "
                        "then the final answer enclosed in <answer>...</answer> tags. The critical answer or key result should be placed within \\boxed{}."
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
    prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase."
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
    parser.add_argument('--model_name_or_path', type=str, default="inclusionAI/M2-Reasoning")
    parser.add_argument('--max_pixels', type=int, default=401408)
    parser.add_argument('--min_pixels', type=int, default=401408)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument("--output_path", type=str,default="/home/yxy1421/tuo/spatial_eval/output/m2_reasoning")
    parser.add_argument("--batch_size", type=int, default=40)
    return parser.parse_args()



#make it batchsize = 50
def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    #load dataset
    dataset =load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")
    #dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks_ViewSpatial-Bench")
    #load m2 reasoning model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name_or_path = os.path.join(args.input_dir, args.model_name_or_path)
    bailing2 = BailingMMInfer(
        args.model_name_or_path, 
        device=device, 
        max_pixels=args.max_pixels, 
        min_pixels=args.min_pixels
    )
    processor = bailing2.processor
    model= bailing2.model
    tokenizer= bailing2.tokenizer



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


                ############
                # print(batch_messages)
                text = [processor.apply_chat_template(conversation=msg, tokenize=False, add_generation_prompt=True, use_system=True) for msg in batch_messages]
                image_inputs, video_inputs = processor.process_vision_info(batch_messages)
                inputs = processor(text=text,images=image_inputs,padding=True,return_tensors="pt")
                # print(inputs)
                # print(tokenizer.decode(inputs['input_ids'][0]))
                inputs = inputs.to(device)
                model.eval()
                # for k in inputs.keys():
                #     if k == "pixel_values" or k == "pixel_values_videos":
                #         inputs[k] = inputs[k].to(dtype=torch.bfloat16)

                
                #############
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(
                            inputs,
                            max_new_tokens=512,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            **bailing2.generation_config,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    batch_output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )

                    print("batch_output_text", batch_output_text)

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