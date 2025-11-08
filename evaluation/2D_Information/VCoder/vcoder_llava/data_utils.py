import nltk
import spacy
from word2number import w2n
import inflect
from num2words import num2words
p = inflect.engine()
import numpy as np
import random

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

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
        except:
            if len(k) >= 3:
                if k == "ski":
                    k = "skis"
                elif k == "gras":
                        k = "grass"
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
                pass

    return num_noun_dict


def _obtain_nouns(gt):
    gt = gt.replace("hair dryer", "hair drier").lower()
    nouns_gt = _get_nouns(gt)

    num_nouns_gt = _get_num_nouns(gt)

    com_keys = []
    for k in nouns_gt.keys():
        if p.plural(k) in num_nouns_gt.keys():
            com_keys.append(k)
    for k in com_keys:
        del nouns_gt[k]

    num_nouns_gt = {**num_nouns_gt, **nouns_gt}
    
    return num_nouns_gt

def generate_qa_pairs(text):
    num_nouns = _obtain_nouns(text)
    qa_pairs = []

    for obj, count in num_nouns.items():
        # Count question
        if count == 1:
            plural_obj = p.plural(obj)
        else:
            plural_obj = obj
        count_question = f"How many {plural_obj} are there in the image?"
        count_answer = f"There {'is' if count == 1 else 'are'} {num2words(count)} {obj} in the image."
        qa_pairs.append((count_question, count_answer))

        prob_positive = np.random.uniform(0,1.)

        if prob_positive > 0.7 or count == 1:
            numeric_presence_question = f"{'Is' if count == 1 else 'Are'} there {num2words(count)} {obj} in the image?"
            numeric_presence_answer = "Yes."
        elif count > 1:
            numbers = [i for i in range(2, count + 6) if i != count]
            # Select a random number from the range
            cnt = random.choice(numbers)
            numeric_presence_question = f"{'Is' if cnt == 1 else 'Are'} there {num2words(cnt)} {obj} in the image?"
            numeric_presence_answer = "No."
        
        qa_pairs.append((numeric_presence_question, numeric_presence_answer))
        random.shuffle(qa_pairs)

    return random.sample(qa_pairs, min(len(qa_pairs), random.choice([1, 2, 3, 4, 5, 6])))

if __name__ == "__main__":

    text = "The objects present in the image are: wall, ceiling, shelf, cabinet, counter, dining table, two people, eighteen bottles, two wine glasses, refrigerator, tv, bowl"
    
    qa = generate_qa_pairs(text)
    from icecream import ic
    ic(qa)
    
