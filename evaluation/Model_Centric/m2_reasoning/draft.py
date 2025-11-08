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
            device_map=self.device_map,
            _attn_implementation="flash_attention_2"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_bos_token=True, trust_remote_code=True,use_fast=False
)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="inclusionAI/M2-Reasoning")
    parser.add_argument('--max_pixels', type=int, default=401408)
    parser.add_argument('--min_pixels', type=int, default=401408)
    parser.add_argument('--max_new_tokens', type=int, default=4096)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name_or_path = os.path.join(args.input_dir, args.model_name_or_path)
    bailing2 = BailingMMInfer(
        args.model_name_or_path, 
        device=device, 
        max_pixels=args.max_pixels, 
        min_pixels=args.min_pixels
    )
    #load image first
    import PIL.Image as Image
    img=Image.open("./assets/example1.png").convert("RGB")
    messages = [
        {
            "role": "system", 
            "content": [
                {"type": "text", "text": "You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. The critical answer or key result should be placed within \\boxed{}."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "image", "image": img},
                {"type": "text", "text": "\nQuestion:\n\nRhombus $QRST$ has an area of 137.9 square meters. If $RT$ is 12.2 meters, find $QS$.\nA. 11.3\nB. 22.4\nC. 22.6\nD. 25.6"},
            ],
        },
    ]
    output_text = bailing2.generate(messages, max_new_tokens=args.max_new_tokens)
    print(output_text)


