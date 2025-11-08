#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m vcoder_llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --question-file ./playground/lmm_datasets/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/lmm_datasets/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/lmm_datasets/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/lmm_datasets/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/lmm_datasets/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-lora
