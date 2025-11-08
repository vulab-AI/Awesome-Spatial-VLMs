#!/bin/bash

python -m vcoder_llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/vcoder_it_llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &

python vcoder_llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/vcoder_it_llava-v1.5-7b-lora.jsonl
