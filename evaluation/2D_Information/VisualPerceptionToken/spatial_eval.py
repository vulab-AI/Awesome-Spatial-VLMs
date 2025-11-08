import torch
import re
from PIL import Image
from transformers import AutoTokenizer
from modeling_qwen2_vl_vpt import VPT_Qwen2VLForConditionalGeneration, VPT_Qwen2VLProcessor
from datasets import load_dataset
import argparse
import os
import json

def resize_max_500(image: Image.Image,max_size=384) -> Image.Image:
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

# ========== 工具函数 ==========
def check_region_tokens(text: str):
    """
    检测 region tokens: <|region_token_start|><|x_0|><|y_1|>...<|region_token_end|>
    """
    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[0-7]\|>)+<\|region_token_end\|>')
    return pattern.findall(text)


def decode_output(output_ids, tokenizer, skip_special=False):
    """解码生成结果"""
    return tokenizer.decode(output_ids[0], skip_special_tokens=skip_special)


# ========== 推理流程 ==========
def extract_region_tokens(answer: str):
    """
    提取 <|region_token_start|> ... <|region_token_end|> 之间的内容（包括边界）。
    如果有多个，返回列表。
    """
    pattern = re.compile(r'<\|region_token_start\|>(?:.*?<\|region_token_end\|>)')
    return pattern.findall(answer)

def run_first_round(model, processor, tokenizer, images, question, device):
    """
    Round1 Prompt 模版:
    user: <question + <image>>
    assistant: ""  (空)
    """
    suffix="Identify the region that can help you answer the question, and then answer the question: "
    # question = question.replace(
    #     "\nIf it's selection question, only return the option letter. Else, only return the answer phrase.", 
    #     ""
    # ).strip()

    prompt = f"<image>\n{question}"
    prompt =  prompt + suffix
    clean_answers=[]
    raw_answers=[]
    output_idss=[]

    for img in images:
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.bfloat16)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9
            )

        raw_answer = decode_output(output_ids, tokenizer, skip_special=False)
        clean_answer = decode_output(output_ids, tokenizer, skip_special=True)

        tokens = extract_region_tokens(clean_answer)
        clean_answer = tokens[0] if tokens else clean_answer

        # clean_answer = extract_region_tokens(clean_answer)[0]
        raw_answers.append(raw_answer)
        clean_answers.append(clean_answer)
        output_idss.append(output_ids)
    

    return raw_answers, clean_answers, output_idss


def run_second_round(model, processor, tokenizer, images, question, answer_1r, device, ground_truth=None):
    """
    Round2 Prompt 模版:
    - clip: <|clip_action_start|><|clip_action|><|clip_action_end|>, <clip_image>
    - dino: <|detection_action_start|><|detection_action|><|detection_action_end|>, <detection_image>
    - sam:  <|seg_action_start|><|seg_action|><|seg_action_end|>, <seg_image>
    - region: <|region_token_start|>...<|region_token_end|>, Region crops -> <image>
    """
    if answer_1r is None:
        print("No Round1 answer provided.")
        return "No answer."
    # actions = {
    #     "clip": "<|clip_action_start|>" in answer_1r,
    #     "dino": "<|detection_action_start|>" in answer_1r,
    #     "sam":  "<|seg_action_start|>" in answer_1r,
    #     "region": bool(check_region_tokens(answer_1r)),
    # }
    actions = {
        "region": any(check_region_tokens(ans) for ans in answer_1r),
    }

    # if not any(actions.values()):
    #     print("No action triggered in Round1 answer.")
    #     return answer_1r  # 没触发 action，直接返回

    # # ====== clip ======
    # if actions["clip"]:
    #     new_prompt = (
    #         f"{question}\n"
    #         "<|clip_action_start|><|clip_action|><|clip_action_end|>\n"
    #         "<clip_image>"
    #     )
    #     inputs = processor(text=new_prompt, images=image, return_tensors="pt").to(device, torch.bfloat16)
    #     with torch.no_grad():
    #         output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
    #     return decode_output(output_ids, tokenizer, skip_special=True)

    # # ====== dino ======
    # if actions["dino"]:
    #     new_prompt = (
    #         f"{question}\n"
    #         "<|detection_action_start|><|detection_action|><|detection_action_end|>\n"
    #         "<detection_image>"
    #     )
    #     inputs = processor(text=new_prompt, images=image, return_tensors="pt").to(device, torch.bfloat16)
    #     with torch.no_grad():
    #         output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
    #     return decode_output(output_ids, tokenizer, skip_special=True)

    # # ====== sam ======
    # if actions["sam"]:
    #     new_prompt = (
    #         f"{question}\n"
    #         "<|seg_action_start|><|seg_action|><|seg_action_end|>\n"
    #         "<seg_image>"
    #     )
    #     inputs = processor(text=new_prompt, images=image, return_tensors="pt").to(device, torch.bfloat16)
    #     with torch.no_grad():
    #         output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
    #     return decode_output(output_ids, tokenizer, skip_special=True)

    # ====== region ======
    
    if actions["region"]:
        image_prompt = "User: "
        for i, img in enumerate(images):
            image_prompt += f"Region {i}: <image>\n"

        answer_1r_all = "Assistant: "
        for i, ans in enumerate(answer_1r):
            answer_1r_all += f"{ans}\n"

        prompt= question + "Identify the region that can help you answer the question, and then answer the question: "
        new_prompt = (
            f"User: {prompt}\n"
            f"{answer_1r_all}"
            f"{image_prompt}"
            "Assistant: Now answer the question. The answer is:"
        )

        # new_prompt = (
        #     f"User: {question}\n"
        #     f"Assistant: {answer_1r}\n"
        #     f"User: Region 0: <image>\n"
        #     f"Assistant: Now answer the question. The answer is:"
        # )

        inputs = processor(text=new_prompt, images=images, return_tensors="pt").to(device, torch.bfloat16)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0, top_p=0.9)
        return decode_output(output_ids, tokenizer, skip_special=True)

    return answer_1r


def run_single_vqa_sample(model, processor, tokenizer, images, question, device):
    """完整流程：Round1 → Round2"""
    raw_ans, clean_ans, output_ids = run_first_round(model, processor, tokenizer, images, question, device)
    # print("Round1 Raw Answer:\n", raw_ans)
    print("\nRound1 Clean Answer:\n", clean_ans)

    final_ans = run_second_round(model, processor, tokenizer, images, question, clean_ans, device)
    print("\nFinal Answer:\n", final_ans)
    return final_ans


# # ========== 主程序 ==========
# if __name__ == "__main__":
#     model_path = "rp-yu/Qwen2-VL-7b-VPT-CLIP"
#     device = "cuda:3" if torch.cuda.is_available() else "cpu"

#     # 加载模型
#     processor = VPT_Qwen2VLProcessor.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#     model = VPT_Qwen2VLForConditionalGeneration.from_pretrained(
#         model_path, torch_dtype=torch.bfloat16
#     ).to(device)

#     # 测试样例
#     image = Image.open("test.jpg").convert("RGB")
#     image = resize_max_500(image)
#     images=[image,image,image,image]  # 多图场景
#     question = "What is the color of the cup on the table?"
#     # print(tokenizer.convert_tokens_to_ids("<|clip_action_start|>"))
#     # print(tokenizer.convert_tokens_to_ids("<|detection_action_start|>"))
#     # print(tokenizer.convert_tokens_to_ids("<|seg_action_start|>"))
#     # print(tokenizer.convert_tokens_to_ids("<|region_token_start|>"))

#     final_answer = run_single_vqa_sample(model, processor, tokenizer, images, question, device)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output/vpt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model_path = "rp-yu/Qwen2-VL-7b-VPT-CLIP"
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    # 加载模型
    processor = VPT_Qwen2VLProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = VPT_Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)

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
            # prompt += "\nIf it's selection question, only return the option letter. Else, only return the answer phrase. "

            # print("Input Prompt:", prompt, flush=True)
            # 推理
            answer = run_single_vqa_sample(model, processor, tokenizer, images, prompt, device)
            
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




