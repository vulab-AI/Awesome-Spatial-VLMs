#!/bin/bash

python -m vcoder_llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --question-file ./playground/lmm_datasets/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/lmm_datasets/eval/vizwiz/test \
    --answers-file ./playground/lmm_datasets/eval/vizwiz/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/lmm_datasets/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/lmm_datasets/eval/vizwiz/answers/llava-v1.5-7b-lora.jsonl \
    --result-upload-file ./playground/lmm_datasets/eval/vizwiz/answers_upload/llava-v1.5-7b-lora.json