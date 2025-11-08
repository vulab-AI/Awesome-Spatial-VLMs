import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_videos,
    tokenizer_special_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os
from PIL import Image
import math
from datasets import load_dataset
import json

def resize_max_500(image: Image.Image,max_size=280) -> Image.Image:
    """将图像最长边缩放到384以内"""
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


# def image_parser(args):
#     out = args.image_file.split(args.sep)
#     return out


# def load_image(image_file):
#     if image_file.startswith("http") or image_file.startswith("https"):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     else:
#         image = Image.open(image_file).convert("RGB")
#     return image


# def load_images(image_files):
#     out = []
#     for image_file in image_files:
#         image = load_image(image_file)
#         out.append(image)
#     return out


def eval_model(args, images):
    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 固定走 image 模式
    mode = 'image'

    # # 加载模型
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name, torch_dtype=torch_dtype
    # )

    # 原始 query
    qs = args.query

    clicks = torch.zeros((0, 3))

    # === 关键改动：根据图片数量插入 image special tokens ===
    num_images = len(images)
    assert num_images >= 1, "images 不能为空：请至少传入一张 PIL.Image"

    # 单个图像 token 串
    one_image_token = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if getattr(model.config, "mm_use_im_start_end", False)
        else DEFAULT_IMAGE_TOKEN
    )

    # 如果用户 query 里已经包含占位符 IMAGE_PLACEHOLDER（通常是 "<image>"）
    if IMAGE_PLACEHOLDER in qs:
        # 将所有占位符替换为正确的 special token（出现几次替换几次）
        qs = qs.replace(IMAGE_PLACEHOLDER, one_image_token)

        # 如果占位符数量少于图片数量，则把剩余的 token 追加到最前面
        placeholder_count = len(re.findall(re.escape(one_image_token), qs))
        if placeholder_count < num_images:
            extra = one_image_token * (num_images - placeholder_count)
            qs = extra + "\n" + qs
        # 如果占位符数量多于图片数量，这里不报错，但会多放 token（通常不建议）
    else:
        # 没写占位符：自动在最前面插入与图片数一致的 token
        qs = (one_image_token * num_images) + "\n" + qs
    # === 关键改动结束 ===

    # 选择对话模板
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3D" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 处理图片到张量（支持多图：images 为 PIL.Image 列表）
    images_tensor = process_images(
        images,
        processor['image'],
        model.config
    ).to(model.device, dtype=torch_dtype)

    depths_tensor = None
    poses_tensor = None
    intrinsics_tensor = None
    clicks_tensor = None  # 图像模式下通常不需要 clicks；若要用可传 clicks.to(...)

    # tokenizer 编码
    input_ids = (
        tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            depths=depths_tensor,
            poses=poses_tensor,
            intrinsics=intrinsics_tensor,
            clicks=clicks_tensor,
            image_sizes=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--video-path", type=str, help="Path to the video file")
    # group.add_argument("--image-file", type=str, help="Path to the image file")

    parser.add_argument("--model-path", type=str, default="ChaimZhu/LLaVA-3D-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--query", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_path", type=str, default="output/llava_3d")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # 加载数据集
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 固定走 image 模式
    mode = 'image'

    # 加载模型
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, torch_dtype=torch_dtype
    )

    image_list = [f"image_{i}" for i in range(32)]
    for bench in dataset.keys():

        all_outputs = []
        print("Processing bench:", bench)
        result_file = os.listdir(args.output_path)
        if f"{bench}_results.json" in result_file:
            print(f"Results for {bench} already exist, skipping...")
            continue

        for item in dataset[bench]:
            # 获取图像
            images = []
            for image_key in image_list:
                image_obj = item.get(image_key, None)
                if image_obj is not None:
                    image = image_obj.convert("RGB")
                    image = resize_max_500(image)
                    images.append(image)

            if len(images) == 0:
                continue
            
            # 构建 prompt
            prompt = item.get("prompt", "")
            prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. Answer:"

            print("Prompt:", prompt)
            args.query = prompt
            answer = eval_model(args,images)

            # 打印结果
            print("Output:", answer, flush=True)
            print("\n", flush=True)

            all_outputs.append({
                "bench_name": bench,
                "id": item["id"],
                "prompt": item["prompt"],
                "options": item["options"],
                "GT": item["GT"],
                "result": answer
            })

            torch.cuda.empty_cache()

        with open(f"{args.output_path}/{bench}_results.json", "w") as f:
            json.dump(all_outputs, f, indent=4)


    # eval_model(args,images)
