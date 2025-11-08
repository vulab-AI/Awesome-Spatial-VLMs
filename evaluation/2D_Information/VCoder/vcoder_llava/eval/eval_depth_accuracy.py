import argparse
from tqdm import tqdm
import nltk
import spacy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

synonyms = open('vcoder_llava/eval/synonyms.txt').readlines()
synonyms = [s.strip().split(', ') for s in synonyms]
WORD_TO_COM = {}
for synonym in synonyms:
    for s in synonym:
        WORD_TO_COM[s] = synonym[0]

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Inference")
    parser.add_argument("--gt_path", type=str, default="path to gt txt files")
    parser.add_argument("--pred_path", type=str, default="path to pred txt files")
    args = parser.parse_args()
    return args

def _obtain_seg_texts(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    
    seg_labels = {}
    for line in lines:
        key = line.split("<IMG>")[1].strip("\n")
        label = line.split("<IMG>")[2].strip("\n")
        seg_labels[key] = label    
    return seg_labels

def extract_conversations(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    seg_preds = {}
    for line in lines:
        if "--------" in line or line.startswith("<<QUESTION>>"):
            continue
        elif line.startswith("Image: "):
            key = line.split("Image: ")[1].strip("\n")
            seg_preds[key] = ""
        else:
            seg_preds[key] = line.strip("<<ANSWER>>: ").strip("\n").split("</s>")[0]
    return seg_preds

def _get_order(lines):
    if len(lines.split(":")) == 1:
        return {}, 0
    lines = lines.split(":")[1]
    doc = nlp(lines)
    nouns = [chunk.text for chunk in doc.noun_chunks]
    order_num = 1
    positions = {}
    for noun in nouns:
        object = noun.split("-")[0].strip()
        if object in WORD_TO_COM.keys():
            object = WORD_TO_COM[object]
        if object not in positions.keys():
            positions[object] = [order_num]
        else:
            positions[object].append(order_num)
        order_num += 1 
    return positions, order_num - 1

def _obtain_object_order(gt, pred):
    gt = gt.replace("hair dryer", "hair drier").lower()
    pred = pred.replace("hair dryer", "hair drier").lower()
    
    position_gt, order_num = _get_order(gt)
    position_pred, _ = _get_order(pred)
    
    return position_gt, position_pred, order_num

def calculate_depth_score(gt_path, pred_path):    
    gt_labels = _obtain_seg_texts(gt_path)
    preds = extract_conversations(pred_path)

    assert all([k in gt_labels.keys() for k in preds.keys()]), "GT and Predicted files don't match!"

    acc_depth_scores = []
    
    for k in tqdm(gt_labels.keys(), total=len(gt_labels.keys())):
        gt = gt_labels[k]
        pred = preds[k]

        position_gt, position_pred, order_num = _obtain_object_order(gt, pred)

        depth_distance = []

        for k in position_gt.keys():
            if position_pred is not None and k in position_pred.keys():
                order_pred = position_pred[k]
                order_gt = position_gt[k]
                if len(order_gt) < len(order_pred):
                    order_gt.extend([100] * (len(order_pred) - len(order_gt)))
                elif len(order_pred) < len(order_gt):
                    order_pred.extend([100] * (len(order_gt) - len(order_pred)))

                for i, j in zip(order_gt, order_pred):
                    if i == 100 and j == 100:
                        continue
                    depth_distance.append(abs(i - j))
            else:
                depth_distance.append(100)
                    
        if len(depth_distance) > 0:
            acc_depth_scores.append(sum(depth_distance) / order_num)

    return acc_depth_scores


if __name__ == "__main__":
    args = parse_args()
    acc_depth_scores = calculate_depth_score(args.gt_path, args.pred_path)
    
    print("Average Depth Score is: {}".format(round((sum(acc_depth_scores) / len(acc_depth_scores)), 2)))