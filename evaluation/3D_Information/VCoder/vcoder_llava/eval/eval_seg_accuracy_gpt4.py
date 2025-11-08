import argparse
from tqdm import tqdm
import os
import nltk
import spacy
from word2number import w2n
import json
import inflect

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

WORD_TO_COM = {
    "man": "person",
    "woman": "person",
    "child": "person",
    "children": "persons",
    "men": "persons",
    "women": "persons",
    "kid": "person",
    "kids": "persons",
    "girl": "person",
    "boy": "person",
    "girls": "persons",
    "boys": "persons",
}

# object names with two words
SPECIAL_WORDS = ['baseball bat',
        'baseball glove',
        'cell phone',
        'dining table',
        'fire hydrant',
        'french fries',
        'hair drier',
        'hot dog',
        'parking meter',
        'potted plant',
        'soccer ball',
        'soccer player',
        'sports ball',
        'stop sign',
        'teddy bear',
        'tennis racket',
        'toy figure',
        'traffic light',
        'wine glass']

p = inflect.engine()

def _remove_specific_word(text, word_to_remove):
    import re
    tokens = re.findall(r'\b\w+\b|[,.]', text)
    result_tokens = []
    word_found = False

    for i, token in enumerate(tokens):
        if token == word_to_remove:
            if not word_found:
                # Keep the first occurrence and mark it as found
                result_tokens.append(token)
                word_found = True
            else:
                # Remove any preceding punctuation if it's just before this word
                if i > 0 and tokens[i-1] in {',', '.'}:
                    result_tokens.pop()
        else:
            result_tokens.append(token)

    # Join tokens and clean up spaces before punctuation
    result_text = ' '.join(result_tokens)
    result_text = re.sub(r'\s([,.](?:\s|$))', r'\1', result_text)
    return result_text

def _get_nouns(lines):
    # function to test if something is a noun
    present_words = []
    for s in SPECIAL_WORDS:
        if s in lines:
            present_words.append(s)
    
    for w in present_words:
        lines = lines.replace(w, "")
    
    is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'NNP'
    # do the nlp stuff
    tokenized = nltk.word_tokenize(lines)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    noun_dict = {}
    if "objects" in nouns:
        nouns.remove("objects")
    if "image" in nouns:
        nouns.remove("image")

    for n in nouns:
        if n in WORD_TO_COM.keys():
            n = WORD_TO_COM[n]
        if n not in noun_dict.keys():
            noun_dict[n] = 1
        else:
            noun_dict[n] += 1
    nouns = {}
    for k, v in noun_dict.items():
        if not (k == "bus" or k == "skis"):
            if v == 1:
                if p.singular_noun(k):
                    k = p.singular_noun(k)
            else:
                if not p.singular_noun(k):
                    k = p.plural(k)
        try:
            w2n.word_to_num(k)
            print("converting {} to number".format(k))
        except:
            if len(k) >= 3:
                if k == "ski":
                    k = "skis"
                nouns[k] = v
    for w in present_words:
        nouns[w] = 1
    return nouns

def _get_num_nouns(lines):
    lines = lines.replace(":", "").replace(".", "")
    doc = nlp(lines)
    num_nouns = [chunk.text for chunk in doc.noun_chunks if any(token.pos_ == 'NUM' for token in chunk)]

    num_noun_dict = {}
    for n in num_nouns:
        nums = n.split(", ")
        for n in nums:
            try:
                w = " ".join(n.split(' ')[1:])
                if w == "ski":
                    w = "skis"
                num_noun_dict[w] = w2n.word_to_num(n.split(' ')[0])
            except:
                print("Error in converting {} to number".format(n))

    return num_noun_dict


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Inference")
    parser.add_argument("--gt_path", type=str, default="path to gt txt files")
    parser.add_argument("--pred_path", type=str, default="path to pred txt files")
    args = parser.parse_args()
    return args

def _obtain_seg_texts(file_path, d_files):
    with open(file_path) as f:
        lines = f.readlines()
    
    seg_labels = {}
    for line in lines:
        key = line.split("<IMG>")[1].strip("\n")
        label = line.split("<IMG>")[2].strip("\n")
        if key not in d_files:
            continue
        label = _remove_specific_word(label, "wall")
        label = _remove_specific_word(label, "window")
        seg_labels[key] = label
    
    return seg_labels

def extract_conversations(file_path, d_files):
    with open(file_path) as f:
        lines = f.readlines()
    # lines = lines[3:-1]
    seg_preds = {}
    for line in lines:
        if "--------" in line or line.startswith("<<QUESTION>>"):
            continue
        elif line.startswith("Image: "):
            key = line.split("Image: ")[1].strip("\n")
            if key not in d_files:
                continue
            seg_preds[key] = ""
        else:
            if key not in d_files:
                continue
            seg_preds[key] = line.strip("<<ANSWER>>: ").strip("\n").split("</s>")[0]
    return seg_preds

def _obtain_nouns(gt, pred):
    gt = gt.replace("hair dryer", "hair drier").lower()
    pred = pred.replace("hair dryer", "hair drier").lower()
    nouns_gt = _get_nouns(gt)
    nouns_pred = _get_nouns(pred)

    num_nouns_gt = _get_num_nouns(gt)
    num_nouns_pred = _get_num_nouns(pred)

    com_keys = []
    for k in nouns_gt.keys():
        if p.plural(k) in num_nouns_gt.keys():
            com_keys.append(k)
    for k in com_keys:
        del nouns_gt[k]
    
    com_keys = []
    for k in nouns_pred.keys():
        if p.plural(k) in num_nouns_pred.keys():
            com_keys.append(k)
    for k in com_keys:
        del nouns_pred[k]

    num_nouns_gt = {**num_nouns_gt, **nouns_gt}
    num_nouns_pred = {**num_nouns_pred, **nouns_pred}
    
    return num_nouns_gt, num_nouns_pred

def calculate_accuracy_hallucination(gt_dir, pred_dir, d_files):
    acc_avg_scores = {}
    hallucination_avg_scores = {}
    json_pred = {}
    json_gt = {}
    for task in ["panoptic"]:
        print("Evaluating for {} segmentation...".format(task))

        json_pred[task] = {}
        json_gt[task] = {}
        
        gt_labels = _obtain_seg_texts(os.path.join(gt_dir, task + ".txt"), d_files)
        preds = extract_conversations(os.path.join(pred_dir, "output_" + task + ".txt"), d_files)

        assert all([k in gt_labels.keys() for k in preds.keys()]), "GT and Predicted files don't match!"

        acc_avg_scores[task] = []
        hallucination_avg_scores[task] = []
        
        for k in tqdm(gt_labels.keys(), total=len(gt_labels.keys())):
            gt = gt_labels[k]
            pred = preds[k]

            num_nouns_gt, num_nouns_pred = _obtain_nouns(gt, pred)

            json_gt[task][k] = num_nouns_gt
            json_pred[task][k] = num_nouns_pred

            acc_scores = []
            hallucination_scores = []

            for k in num_nouns_gt.keys():
                if num_nouns_pred is not None and k in num_nouns_pred.keys():
                    score = min(num_nouns_gt[k], num_nouns_pred[k]) / max(num_nouns_gt[k], num_nouns_pred[k])
                    acc_scores.append(score)
                else:
                    acc_scores.append(0.0)
                        
            if len(acc_scores) > 0:
                acc_avg_scores[task].append(sum(acc_scores) / len(acc_scores))
            
            for k in num_nouns_pred.keys():
                if num_nouns_gt is not None and k in num_nouns_gt.keys():
                    score = min(num_nouns_gt[k], num_nouns_pred[k]) / max(num_nouns_gt[k], num_nouns_pred[k])
                    hallucination_scores.append(1.0 - score)
                else:
                    hallucination_scores.append(1.0)
                        
            if len(hallucination_scores) > 0:
                hallucination_avg_scores[task].append(sum(hallucination_scores) / len(hallucination_scores))
        
    with open(f"{gt_dir}/gt.json", "w") as outfile:
        json.dump(json_gt, outfile)
    
    with open(f"{pred_dir}/pred.json", "w") as outfile:
        json.dump(json_pred, outfile)

    return acc_avg_scores, hallucination_avg_scores


if __name__ == "__main__":
    args = parse_args()

    gt_dir = args.gt_path
    pred_dir = args.pred_path

    with open("done_ims.txt") as f:
        lines = f.readlines()
    
    d_files = []
    for l in lines:
        d_files.append(l.strip("\n").split("/")[-1])
    
    acc_avg_scores, hallucination_avg_scores = calculate_accuracy_hallucination(gt_dir, pred_dir, d_files)

    for k, v in acc_avg_scores.items():
        print("Average accuracy for {} segmentation is: {}".format(k, round((sum(v) / len(v))*100, 1)))
        print("Average hallucination for {} segmentation is: {}".format(k, round((sum(hallucination_avg_scores[k]) / len(hallucination_avg_scores[k]))*100, 1)))
        print('-----------------------------------------')