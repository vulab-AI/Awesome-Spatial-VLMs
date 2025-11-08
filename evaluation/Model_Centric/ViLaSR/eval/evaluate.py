import numpy as np
import json
import argparse
import re
import torch
from tqdm import tqdm
from utils.cal_metric_vsibench import calculate_average_scores_vsibench
from utils.cal_metric_sparbench import calculate_average_scores_sparbench

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error <= (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()

def compute_vci_score(output_ans, gt_ans):
    ACTION_PAIRS = {
        "move_right_left": ("move_right", "move_left"),
        "move_up_down": ("move_up", "move_down"),
        "move_forward_backward": ("move_forward", "move_backward"),
        "rotate_right_left": ("rotate_right", "rotate_left"),
        "rotate_up_down": ("rotate_up", "rotate_down")
    }
    
    try:
        answer_dict = parse_instruction(output_ans)
        gt_dict = parse_instruction(gt_ans)
        
        answer_list = []
        gt_list = []
        
        for action_pair, (pos, neg) in ACTION_PAIRS.items():
            net_pred = answer_dict.get(pos, 0) - answer_dict.get(neg, 0)
            net_gt = gt_dict.get(pos, 0) - gt_dict.get(neg, 0)
            answer_list.append(net_pred)
            gt_list.append(net_gt)
            
        mra_list = [
            mean_relative_accuracy(answer, gt)
            for gt, answer in zip(gt_list, answer_list)
        ]
        
        return np.mean(mra_list)
        
    except Exception as e:
        print(f"Error in VCI score calculation: {e}, output: {output_ans}")
        return 0.0

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

def reward_fn(model_output, gt_ans, question_type):
    output_ans = extract_answer(model_output)
    gt_ans = extract_answer(gt_ans)
    if question_type == "multiple choice":
        return 1.0 if output_ans.strip()[0].lower() == gt_ans.strip()[0].lower() else 0.0
    elif question_type == "numerical":
        gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
        out_has_decimal = ("." in output_ans) or ("," in output_ans)
        if gt_has_decimal != out_has_decimal:
            return 0.0
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
    elif question_type == "regression":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        mra = mean_relative_accuracy(out_number, gt_number)
        return mra
    elif question_type == "vci":
        return compute_vci_score(output_ans, gt_ans)
    else:
        return 0.0

def score_fixed_answer(args):
    scores = []
    results = []
    pred_answers = [json.loads(q) for q in open(args.result_file)]

    print("Length: ", len(pred_answers))
    for input_example in pred_answers:
        pred_answer_text = input_example.get('model_output', "")
        gt_answer_text = input_example.get('answer', "")
        problem_type = input_example.get("problem_type", "")
        result = input_example.copy()

        score = reward_fn(pred_answer_text,gt_answer_text, problem_type)

        result['score'] = score
        results.append(result)
        scores.append(score)

    print('The avg score is: %f' % np.mean(scores))

    with open(args.output_result, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    if args.dataset == 'vsi_bench':
        calculate_average_scores_vsibench(results)
    elif args.dataset == 'spar_bench':
        calculate_average_scores_sparbench(results)
    print(f"Results have been saved to {args.output_result} in JSONL format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--result-file', type=str)
    args = parser.parse_args()
    score_fixed_answer(args)
