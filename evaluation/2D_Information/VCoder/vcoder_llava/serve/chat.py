"""
A model worker executes the model.
"""
import argparse
import json
import torch

from vcoder_llava.utils import server_error_msg
from vcoder_llava.model.builder import load_pretrained_model
from vcoder_llava.mm_utils import process_images, load_image_from_base64, tokenizer_seg_token, tokenizer_depth_seg_token, tokenizer_image_token, KeywordsStoppingCriteria
from vcoder_llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    SEG_TOKEN_INDEX, DEFAULT_SEG_TOKEN,
    DEPTH_TOKEN_INDEX, DEFAULT_DEPTH_TOKEN,
)
from transformers import TextIteratorStreamer
from threading import Thread

class Chat:
    def __init__(self, model_path, model_base, model_name,
                 load_8bit, load_4bit, device, logger):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} ...")
        self.tokenizer, self.model, self.image_processor, self.seg_image_processor, self.depth_image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = 'llava' in self.model_name.lower()
        self.is_seg = "vcoder" in self.model_name.lower()
        self.is_depth = "ds" in self.model_name.lower()

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor, seg_image_processor, depth_image_processor = self.tokenizer, self.model, self.image_processor, self.seg_image_processor, self.depth_image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        segs = params.get("segs", None)
        depths = params.get("depths", None)
        num_image_tokens = 0
        num_seg_tokens = 0
        num_depth_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)
                
                replace_token = DEFAULT_IMAGE_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches

                if segs is not None and len(segs) > 0 and self.is_seg:
                    if len(segs) != prompt.count(DEFAULT_SEG_TOKEN):
                        raise ValueError("Number of segs does not match number of <seg> tokens in prompt")
                    
                    segs = [load_image_from_base64(seg) for seg in segs]
                    segs = process_images(segs, seg_image_processor, model.config)
                
                    if type(segs) is list:
                        segs = [seg.to(self.model.device, dtype=torch.float16) for seg in segs]
                    else:
                        segs = segs.to(self.model.device, dtype=torch.float16)
                    
                    replace_seg_token = DEFAULT_SEG_TOKEN
                    prompt = prompt.replace(DEFAULT_SEG_TOKEN, replace_seg_token)
                    num_seg_tokens = prompt.count(replace_seg_token) * model.get_vision_tower().num_patches

                    if depths is not None and len(depths) > 0 and self.is_depth:
                        if len(depths) != prompt.count(DEFAULT_DEPTH_TOKEN):
                            raise ValueError("Number of depths does not match number of <depth> tokens in prompt")
                        
                        depths = [load_image_from_base64(depth) for depth in depths]
                        depths = process_images(depths, depth_image_processor, model.config)
                    
                        if type(depths) is list:
                            depths = [depth.to(self.model.device, dtype=torch.float16) for depth in depths]
                        else:
                            depths = depths.to(self.model.device, dtype=torch.float16)
                        
                        replace_depth_token = DEFAULT_DEPTH_TOKEN
                        prompt = prompt.replace(DEFAULT_DEPTH_TOKEN, replace_depth_token)
                        num_depth_tokens = prompt.count(replace_depth_token) * model.get_vision_tower().num_patches
                    else:
                        depths = None
                else:
                    segs = None
                    depths = None
            else:
                images = None
                segs = None
                depths = None
            image_args = {"images": images, "segs": segs, "depths": depths}
        else:
            images = None
            segs = None
            depths = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        if self.is_seg and segs is not None:
            if self.is_depth and depths is not None:
                input_ids = tokenizer_depth_seg_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, DEPTH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            else:
                input_ids = tokenizer_seg_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        else:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens - num_seg_tokens - num_depth_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        generated_text = model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        )
        # thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()