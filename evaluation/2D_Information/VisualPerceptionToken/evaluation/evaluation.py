import os, yaml, json, sys, copy, re, torch, argparse
import subprocess
from PIL import Image

from gpt import bulk_evaluate

Action_tokens = {
    "region_x": "<|x_0|>,<|x_1|>,<|x_2|>,<|x_3|>,<|x_4|>,<|x_5|>,<|x_6|>,<|x_7|>".split(","),
    "region_y": "<|y_0|>,<|y_1|>,<|y_2|>,<|y_3|>,<|y_4|>,<|y_5|>,<|y_6|>,<|y_7|>".split(","),
    "dino": "<|detection_action_start|>",
    "clip": "<|clip_action_start|>",
    "sam": "<|seg_action_start|>",
}

evaluation_prompt = "You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form 'score: <score>'. The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.\n"
"""
 Question: { }
 Standard answer: { }
 Model`s answer: { }
 """

os.chdir("/root/LLaMA-Factory")

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def check_question(meta_item, generated_item):
    meta_question = meta_item["messages"][0]["content"]
    meta_text = meta_question.replace("<image>", "").replace("\n", "")
    meta_text = meta_text.replace("Identify the region that can help you answer the question, and then answer the question","")
    meta_text = meta_text.replace("Require additional perception features, and then answer the question","")
    meta_text = meta_text.replace("Answer the question using a single word or phrase","")
    meta_text = meta_text.replace("Answer with the option's letter from the given choices directly","")
    meta_text = meta_text.strip(". ")

    generated_question = generated_item["prompt"]
    generated_question = generated_question.replace("<image>", "").replace("\n", "")
    if meta_text not in generated_question:
        raise Exception(f"Meta text not in generated question: {meta_text} not in {generated_question}")

def check_region_tokens(text):

    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[01234567]\|>)+<\|region_token_end\|>')
    matches = pattern.finditer(text)

    found_tokens = []
    for match in matches:
        match_str = match.group()
        # match_str = match_str.replace("<|region_token_start|>","").replace("<|region_token_end|>","")
        match_tokens_x = [token for token in Action_tokens["region_x"] if token in match_str]
        match_tokens_y = [token for token in Action_tokens["region_y"] if token in match_str]
        found_tokens.append((match_tokens_x, match_tokens_y))

    if found_tokens:
        return found_tokens, True
    else:
        return None, False

def generate_dino_item(gt_item, index):
    image_path = gt_item["images"][0]
    new_message = [
        gt_item["messages"][0],
        {"role": "assistant", "content": "<|detection_action_start|><|detection_action|><|detection_action_end|>"},
        {"role": "user", "content": "<detection_image>"},
        gt_item["messages"][-1],
    ]
    new_item = {
        "index": index,
        "messages": new_message,
        "images": [image_path],
        "detection_images": [image_path],
        "clip_images": [],
        'seg_images': [],
    }
    return new_item

def generate_clip_item(gt_item, index):
    image_path = gt_item["images"][0]
    new_message = [
        gt_item["messages"][0],
        {"role": "assistant", "content": "<|clip_action_start|><|clip_action|><|clip_action_end|>"},
        {"role": "user", "content": "<clip_image>"},
        gt_item["messages"][-1],
    ]
    new_item = {
        "index": index,
        "messages": new_message,
        "images": [image_path],
        "detection_images": [],
        "clip_images": [image_path],
        'seg_images': [],
    }
    return new_item

def generate_sam_item(gt_item, index):
    image_path = gt_item["images"][0]
    new_message = [
        gt_item["messages"][0],
        {"role": "assistant", "content": "<|seg_action_start|><|seg_action|><|seg_action_end|>"},
        {"role": "user", "content": "<seg_image>"},
        gt_item["messages"][-1],
    ]
    new_item = {
        "index": index,
        "messages": new_message,
        "images": [image_path],
        "detection_images": [],
        "clip_images": [],
        'seg_images': [image_path],
    }
    return new_item

def from_region_tokens_to_region(region_tokens):
    match_tokens_x, match_tokens_y = region_tokens
    x_indices = [int(item.replace("<|x_","").replace("|>","")) for item in match_tokens_x]
    y_indices = [int(item.replace("<|y_","").replace("|>","")) for item in match_tokens_y]
    minx = min(x_indices)
    miny = min(y_indices)
    maxx = max(x_indices)
    maxy = max(y_indices)
    region = [(minx, miny), (maxx, maxy)]
    return region

def region_token_to_new_image_path(found_region_tokens, meta_item, num_cut=8):
    regions = from_region_tokens_to_region(found_region_tokens)
    max_x = max([x for x,y in regions])
    min_x = min([x for x,y in regions])
    max_y = max([y for x,y in regions])
    min_y = min([y for x,y in regions])

    max_x = min(max_x+1, num_cut-1)
    min_x = max(min_x-1, 0)
    max_y = min(max_y+1, num_cut-1)
    min_y = max(min_y-1, 0)

    image_path = meta_item["images"][0]
    image = Image.open(image_path)
    width, height = image.size
    st_w = 0
    st_h = 0

    grid_width = width / num_cut
    grid_height = height / num_cut

    new_x_min = st_w + (min_x ) * grid_width
    new_y_min = st_h + (min_y ) * grid_height
    new_x_max = st_w + (max_x + 1) * grid_width
    new_y_max = st_h + (max_y + 1) * grid_height

    new_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
    bbox_subfix = f"_{int(new_bbox[0])},{int(new_bbox[1])},{int(new_bbox[2])},{int(new_bbox[3])}"

    new_image_path = image_path + bbox_subfix
    return new_image_path

def generate_region_item(gt_item, generated_answer, index, ls_found_region_tokens, num_cut=8):

    ls_new_image_path = [region_token_to_new_image_path(found_region_tokens, gt_item, num_cut=8) for found_region_tokens in ls_found_region_tokens]
    
    if len(ls_new_image_path) == 1:
        new_image_prompt = "<image>"
    else:
        new_image_prompt = ""
        for i in range(len(ls_new_image_path)):
            new_image_prompt += f"Region {i}: <image>\n"

    new_message = [
        gt_item["messages"][0],
        {"role": "assistant", "content": generated_answer},
        {"role": "user", "content": new_image_prompt },
        {"role": "assistant", "content": gt_item["messages"][-1]["content"]}
    ]
    new_item = {
        "index": index,
        "messages": new_message,
        "images": [gt_item["images"][0]] + ls_new_image_path,
        "detection_images": [],
        "clip_images": [],
        'seg_images': [],
    }
    return new_item

def generate_sec_round_data_items(original_data_file_path, answer_1r_file_path):
    data = json.load(open(original_data_file_path, "r"))
    answer_1r = [json.loads(l) for l in open(answer_1r_file_path, "r").readlines()]

    new_data = []
    finished_data = []

    for i, (gt_item, generated_item) in enumerate(zip(data[:1000], answer_1r[:1000])):
        geneated_answer = generated_item["predict"]
        do_dinoo = Action_tokens["dino"] in geneated_answer
        do_clip = Action_tokens["clip"] in geneated_answer
        do_sam = Action_tokens["sam"] in geneated_answer
        ls_found_region_tokens, do_region = check_region_tokens(geneated_answer)
        if [do_dinoo, do_clip, do_sam, do_region].count(True) > 1:
            finished_data.append({
                "index": i,
                "question": gt_item["messages"][0]["content"],
                "answer": geneated_answer,
                "ground_truth": gt_item["messages"][-1]["content"].replace("Now answer the question.\n", ""),
                "images": gt_item["images"]
            })
        elif do_dinoo:
            new_item = generate_dino_item(gt_item, i)
            new_data.append(new_item)
            # finished_item = None
        elif do_clip:
            new_item = generate_clip_item(gt_item, i)
            new_data.append(new_item)
            # finished_item = None
        elif do_sam:
            new_item = generate_sam_item(gt_item, i)
            new_data.append(new_item)
            # finished_item = None
        elif do_region:
            try:
                new_item = generate_region_item(gt_item, geneated_answer, i, ls_found_region_tokens)
                # finished_item = None
                new_data.append(new_item)
            except Exception as e:
                print(e, generated_item["predict"])
        else:
            # raise Exception("Neither dino nor region tokens found in the answer")
            finished_data.append({
                "index": i,
                "question": gt_item["messages"][0]["content"],
                "answer": geneated_answer,
                "ground_truth": gt_item["messages"][-1]["content"].replace("Now answer the question.\n", ""),
                "images": gt_item["images"]
            })
        
    
    return new_data, finished_data


def generate_first_round_data_items(original_data_file_path):
    data = json.load(open(original_data_file_path, "r"))
    new_data = []
    for i, meta_item in enumerate(data[:1000]):
        messages = [
            {
            "content": meta_item["messages"][0]["content"], 
            "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            },
        ]
        new_item = {
            "index": i,
            "messages": messages,
            "images": meta_item["images"][:1],
            "detection_images": [],
            "clip_images": [],
            'seg_images': [],
        }
        new_data.append(new_item)
    return new_data
    
def split_image_path(current_query_item):
    images = current_query_item["images"]
    messages = current_query_item["messages"]
    split_images = []
    finished_images = 0
    for i in range(len(messages)//2):
        index = i*2
        message = messages[index]["content"]
        num_image = message.count("<image>")
        images_this_round = []
        for j in range(finished_images, finished_images+num_image):
            images_this_round.append(images[j])
        finished_images += num_image
        split_images.append(images_this_round)
    return split_images

def extract_score(score_str):
    try:
        pattern = re.compile(r'[01]\.?\d*')
        match = pattern.search(score_str)
        score = float(match.group())
    except Exception as e:
        score = 0
    return score

base_yaml_file = "evaluation.yaml"
base_parameters = read_yaml(base_yaml_file)

temp_dataset_name = "vpt_evaluation"
temp_dataset_file = "evaluation.json"

if __name__ == "__main__":

    datasets = [
        "DocVQA_region_test_8cut",
        "TextVQA_region_test_8cut",
        "TextCap_region_test_8cut",
        "DUDE_region_test",

        "GQA_region_test",
        "OI_region_test",
        "VSR_region_test",

        "CUB_Birds_action_test",
        "LLaVA_COCO_free_action_test",
        "LLaVA_COCO_single_action_test",
        "Flickr30k_action_test",
        "POPE_action_test",
    ]

    models = {
        "models/Qwen2-VL-2b-VPT-CLIP":{
            "num_inner_forward_run":2,
            "vision_encoder_ls":"clip",
        },
        "models/Qwen2-VL-2b-VPT-Det":{
            "num_inner_forward_run":2,
            "vision_encoder_ls":"dino",
        },
        "models/Qwen2-VL-2b-VPT-Seg":{
            "num_inner_forward_run":2,
            "vision_encoder_ls":"sam",
        },
        "models/Qwen2-VL-7b-VPT-CLIP":{
            "num_inner_forward_run":2,
            "vision_encoder_ls":"clip",
        },
    }

    p_val,k_val = 0.1,10

    for dataset_name in datasets:
        for model_path, model_para in models.items():
            # overall config
            output_dir = os.path.join(model_path, dataset_name, f"p{p_val}_k{k_val}")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir+ "/round1", exist_ok=True)
            os.makedirs(output_dir+ "/round2", exist_ok=True)

            if os.path.exists(output_dir+"/final_scores.txt"):
                with open(output_dir+"/final_scores.txt", "r") as score_file:
                    score_str = score_file.read()
                print(output_dir+"/final_scores.txt Finished. ", score_str)
                continue

            dataset_file = f"../datasets/{dataset_name}.json"
            image_resolution = 512
            gpt_evaluation_batch_size = 8

            # data 1r
            data_1r = generate_first_round_data_items(dataset_file)
            json.dump(
                data_1r, 
                open(temp_dataset_file, "w"),
                indent=2
            )

            # generation 1r
            parameters_copy = copy.deepcopy(base_parameters)
            parameters_copy["output_dir"] = output_dir + "/round1"
            parameters_copy["model_name_or_path"] = model_path
            parameters_copy["eval_dataset"] = temp_dataset_name
            parameters_copy["image_resolution"] = image_resolution
            parameters_copy["top_k"] = k_val
            parameters_copy["top_p"] = p_val
            for k,v in model_para.items():
                if v is not None:
                    parameters_copy[k] = v

            log_file = output_dir+f"/round1/generation.log"

            command = "cd /root/LLaMA-Factory; llamafactory-cli train "
            for k,v in parameters_copy.items():
                command = command + f"--{k} {v} "
            command = command + f" > {log_file} 2>&1"
            if not os.path.exists(output_dir + "/round1/generated_predictions.jsonl"):
                subprocess.run(command, shell=True, check=True)

            # data 2r
            # read the output file
            output_file_1r = output_dir + "/round1/generated_predictions.jsonl" 

            data_2r, final_answers = generate_sec_round_data_items(dataset_file, output_file_1r)
            if len(data_2r) == 0:
                    pass
            else:
                json.dump(
                    data_2r, 
                    open(temp_dataset_file, "w"),
                    indent=2
                )
                # break
                # generation 2r
                parameters_copy = copy.deepcopy(base_parameters)
                parameters_copy["output_dir"] = output_dir + "/round2"
                parameters_copy["model_name_or_path"] = model_path
                parameters_copy["eval_dataset"] = temp_dataset_name
                parameters_copy["image_resolution"] = image_resolution
                parameters_copy["top_k"] = k_val
                parameters_copy["top_p"] = p_val
                parameters_copy["per_device_eval_batch_size"] = 4
                for k,v in model_para.items():
                    if v is not None:
                        parameters_copy[k] = v
                        
                log_file = output_dir+f"/round2/generation.log"

                command = "cd /root/LLaMA-Factory; llamafactory-cli train "
                for k,v in parameters_copy.items():
                    command = command + f"--{k} {v} "
                command = command + f" > {log_file} 2>&1"
                if not os.path.exists(output_dir + "/round2/generated_predictions.jsonl"):
                    subprocess.run(command, shell=True, check=True)

            # evaluation
            output_file_2r = output_dir + "/round2/generated_predictions.jsonl" 

            with open(dataset_file, "r") as gt_data_file:
                gt_data = json.load(gt_data_file)
                with open(output_file_2r, "r") as output_file:
                    for i, (data_2r_item, generated_line) in enumerate(zip(data_2r, output_file.readlines())):
                        question_index = data_2r_item["index"]
                        gt_item = gt_data[question_index]

                        generated_item = json.loads(generated_line)

                        check_question(gt_item, generated_item)

                        answer = generated_item["predict"]

                        final_answers.append({
                            "index": question_index,
                            "question": gt_item["messages"][0]["content"],
                            "answer": answer,
                            "ground_truth": gt_item["messages"][-1]["content"].replace("Now answer the question.\n", ""),
                            "images": gt_item["images"]
                        })

            # evaluate the final answers
            final_scores = bulk_evaluate(final_answers, gpt_evaluation_batch_size, evaluation_prompt)
            # print(final_scores)

            for score_idx, (question_idx, score_str) in enumerate(final_scores):
                assert question_idx == final_answers[score_idx]["index"]
                final_answers[score_idx]["score"] = score_str

            with open(output_dir+"/final_answers.json", "w") as final_answer_file:
                json.dump(final_answers, final_answer_file, indent=2)

            ls_scores = []
            for qidx, score_str in final_scores:
                score_val = extract_score(score_str)
                ls_scores.append(score_val)

            print(output_dir)
            print(f"final scores: {sum(ls_scores)/len(ls_scores)}")
            with open(output_dir+"/final_scores.txt", "w") as score_file:
                score_file.write(f"final scores: {sum(ls_scores)/len(ls_scores)}")