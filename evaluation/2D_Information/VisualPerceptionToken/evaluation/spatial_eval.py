import os, json, copy, subprocess, yaml, re
from PIL import Image

# ====== 基础工具 ======
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_region_tokens(text):
    # 检测 <|x_i|><|y_j|> token
    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[0-7]\|>)+<\|region_token_end\|>')
    return pattern.findall(text)

# ====== Round1: 构造输入并推理 ======
def run_first_round(images, texts, base_parameters, model_path, output_dir):
    # 构造一次推理输入
    messages = [
        {"content": texts[0], "role": "user"},
        {"content": "", "role": "assistant"},
    ]
    meta_item = {
        "index": 0,
        "messages": messages,
        "images": images,
        "detection_images": [],
        "clip_images": [],
        "seg_images": [],
    }
    temp_dataset_file = "evaluation.json"
    json.dump([meta_item], open(temp_dataset_file, "w"), indent=2)

    # 配置参数
    parameters_copy = copy.deepcopy(base_parameters)
    parameters_copy.update({
        "output_dir": output_dir + "/round1",
        "model_name_or_path": model_path,
        "eval_dataset": "vpt_evaluation",
        "image_resolution": 512,
        "top_k": 10,
        "top_p": 0.1,
    })

    # 运行一次推理
    os.makedirs(output_dir + "/round1", exist_ok=True)
    log_file = output_dir + "/round1/generation.log"
    command = "cd /home/tuo/Codes/VisualPerceptionToken/LLaMA-Factory; llamafactory-cli train "
    for k, v in parameters_copy.items():
        command += f"--{k} {v} "
    command += f"> {log_file} 2>&1"
    subprocess.run(command, shell=True, check=True)

    # 读取模型输出
    output_file = output_dir + "/round1/generated_predictions.jsonl"
    with open(output_file) as f:
        line = json.loads(f.readline())
    return line["predict"]

# ====== Round2: 根据 Action_tokens 再推理 ======
def run_second_round(images, texts, answer_1r, base_parameters, model_path, output_dir):
    # 检查动作
    do_dino = "<|detection_action_start|>" in answer_1r
    do_clip = "<|clip_action_start|>" in answer_1r
    do_sam  = "<|seg_action_start|>" in answer_1r
    do_region = bool(check_region_tokens(answer_1r))

    if not any([do_dino, do_clip, do_sam, do_region]):
        return answer_1r  # 没有触发动作，直接用 Round1 答案

    # 构造二轮输入（示例只写 clip，其他类似）
    if do_clip:
        new_message = [
            {"role": "user", "content": texts[0]},
            {"role": "assistant", "content": "<|clip_action_start|><|clip_action|><|clip_action_end|>"},
            {"role": "user", "content": "<clip_image>"},
            {"role": "user", "content": texts[-1]},
        ]
        meta_item = {
            "index": 0,
            "messages": new_message,
            "images": images,
            "detection_images": [],
            "clip_images": images,
            "seg_images": [],
        }
    else:
        return answer_1r  # 为简单起见，先不展开 dino/sam/region

    temp_dataset_file = "evaluation.json"
    json.dump([meta_item], open(temp_dataset_file, "w"), indent=2)

    # 配置参数
    parameters_copy = copy.deepcopy(base_parameters)
    parameters_copy.update({
        "output_dir": output_dir + "/round2",
        "model_name_or_path": model_path,
        "eval_dataset": "vpt_evaluation",
        "image_resolution": 512,
        "top_k": 10,
        "top_p": 0.1,
        "per_device_eval_batch_size": 4,
    })

    os.makedirs(output_dir + "/round2", exist_ok=True)
    log_file = output_dir + "/round2/generation.log"
    command = "cd /home/tuo/Codes/VisualPerceptionToken/LLaMA-Factory; llamafactory-cli train "
    for k, v in parameters_copy.items():
        command += f"--{k} {v} "
    command += f"> {log_file} 2>&1"
    subprocess.run(command, shell=True, check=True)

    # 读取最终输出
    output_file = output_dir + "/round2/generated_predictions.jsonl"
    with open(output_file) as f:
        line = json.loads(f.readline())
    return line["predict"]

# ====== 主流程（单条数据） ======
def run_single_vqa_sample(images, texts, model_path):
    # 加载参数和数据
    base_parameters = read_yaml("evaluation.yaml")

    output = "eval_output"
    os.makedirs(output, exist_ok=True)

    # Round 1
    ans1 = run_first_round(images, texts, base_parameters, model_path, output)
    print("Round1 Answer:", ans1)

    # Round 2 (如果有 Action_tokens)
    final_ans = run_second_round(images, texts, ans1, base_parameters, model_path, output)
    print("Final Answer:", final_ans)

    return final_ans


# ====== 调用 ======
model_path = "rp-yu/Qwen2-VL-7b-VPT-CLIP"
images = ['test.jpg']
texts = ["What is the color of the hair of the girl?"]

final_answer = run_single_vqa_sample(images, texts, model_path)
