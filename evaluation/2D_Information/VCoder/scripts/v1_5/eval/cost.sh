#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="vcoder_llava-v1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m vcoder_llava.eval.model_seg_loader \
        --model-path shi-labs/$CKPT \
        --image-folder ./playground/data/coco/val2017 \
        --seg-image-folder ./playground/data/coco_segm_text/val \
        --output-file ./playground/data/eval/seg/$CKPT/output \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use_seg &
done

wait

semantic_output_file=./playground/data/eval/seg/$CKPT/output_semantic.txt
instance_output_file=./playground/data/eval/seg/$CKPT/output_instance.txt
panoptic_output_file=./playground/data/eval/seg/$CKPT/output_panoptic.txt

# Clear out the output files if it exists.
> "$semantic_output_file"
> "$instance_output_file"
> "$panoptic_output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seg/$CKPT/output_semantic_${CHUNKS}_${IDX}.txt >> "$semantic_output_file"
    cat ./playground/data/eval/seg/$CKPT/output_instance_${CHUNKS}_${IDX}.txt >> "$instance_output_file"
    cat ./playground/data/eval/seg/$CKPT/output_panoptic_${CHUNKS}_${IDX}.txt >> "$panoptic_output_file"
done

python -m vcoder_llava.eval.eval_seg_accuracy \
    --gt_path "./playground/data/coco_segm_text/val/" \
    --pred_path "./playground/data/eval/seg/$CKPT/"
