#!/bin/bash

python -m vcoder_llava.eval.model_vqa_mme \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --question-file ./playground/lmm_datasets/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/lmm_datasets/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/lmm_datasets/eval/MME/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/lmm_datasets/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-lora

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-lora
