import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path, tokenizer_special_token

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    results = dict()
    for line in tqdm(questions):
        prompt_id = line["prompt_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line['conversations'][0]['value']
        answers = line['conversations'][1]['value']
        cur_prompt = qs
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs = qs.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
        boxes_seq = line['conversations'][0].get('boxes_seq', None)

        if 'target' in line:
            boxes = line['target']['boxes']
            clicks = []
            for box in boxes:
                click = [round(coord, 3) for coord in box[:3]]
                clicks.append(click)  # list of list 
            clicks = torch.tensor(clicks)
            # print('bboxes num:', len(boxes), 'prompt_id:', prompt_id, 'clicks shape:', clicks.shape)
            objs_num = len(boxes)
            obj_placeholder = '<boxes>, '
            objs_str = obj_placeholder * objs_num
            objs_str = objs_str.rstrip(', ')
            qs = qs.replace('<boxes>', objs_str)
        else:
            clicks = torch.zeros((0, 3))

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()

        videos_dict = process_videos(
            video_path,
            processor['video'],
            mode='random',
            device=model.device,
            text=cur_prompt
        )

        images_tensor = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
        depths_tensor = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
        poses_tensor = videos_dict['poses'].to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)
        clicks_tensor = clicks.to(model.device, dtype=torch.bfloat16)

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
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        result = dict(question=cur_prompt, pred=[outputs], gt=answers)
        # print(result)
        results[prompt_id] =result

    with open(answers_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-region-voxelize-1-50-mmscan-4gpu-p-encoder")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_mmscan_QA_val_v2.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_mmscan_val_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
