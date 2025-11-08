#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="vcoder_ds_llava-v1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m vcoder_llava.eval.model_depth_loader \
        --model-path shi-labs/$CKPT \
        --image-folder ./playground/data/coco/val2017 \
        --seg-image-folder ./playground/data/coco_segm_text/val/ \
        --depth_image-folder ./playground/data/coco_segm_text/depth/val/depth \
        --output-file ./playground/data/eval/depth/$CKPT/output_depth \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use_depth_seg &
done

wait

output_file=./playground/data/eval/depth/$CKPT/output_depth.txt

# Clear out the output files if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/depth/$CKPT/output_depth_${CHUNKS}_${IDX}.txt >> "$output_file"
done

python -m vcoder_llava.eval.eval_depth_accuracy \
    --gt_path "./playground/data/coco_segm_text/depth/val/panoptic_order.txt" \
    --pred_path "./playground/data/eval/depth/$CKPT/output_depth.txt"
