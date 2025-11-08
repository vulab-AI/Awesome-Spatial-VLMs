import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os

import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ====== 全局模型加载 ======
disable_torch_init()
MODEL_PATH = "/home/tuo/Codes/survey_eval/Aurora-perception/checkpoints/train_depth_annealing-llava-v1.5-13b-task-lora"   # 修改成你的模型路径
MODEL_NAME = get_model_name_from_path(MODEL_PATH)

print(">>> Loading model ...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, "liuhaotian/llava-v1.5-13b", MODEL_NAME
)
model.cuda().eval()
print(">>> Model loaded.")


def resize_max_500(image: Image.Image) -> Image.Image:
    """将图像最长边缩放到384以内"""
    max_size = 384
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


def run_inference(images, question, conv_mode="llava_v1",
                  temperature=0, top_p=None, num_beams=1):
    """
    单次推理函数
    """
    # 构建 prompt
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 文本 tokens
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # 图像预处理（兼容单图/多图/anyres）
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):  # anyres 情况
        image_tensor = torch.cat(image_tensor, dim=0)
    elif image_tensor.ndim == 3:  # (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

    # 推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=256,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output/llava_aurora")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # 加载数据集
    dataset = load_dataset("LLDDSS/Awesome_Spatial_VQA_Benchmarks")

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
            answer = run_inference(images, prompt, temperature=0)

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
