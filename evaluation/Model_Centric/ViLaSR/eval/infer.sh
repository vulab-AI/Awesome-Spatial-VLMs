#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
split=0
all=1
DATADIR="./benchmark/"
script_paths=(  
"vsi_bench"
"maze"
"SpatialEval_spatialreal"
"spar_bench"
"mmsi_bench"
)

model_name=Qwen2.5_VL_7B
CKPT=inclusionAI/ViLaSR                 # zero_shot/cold_start/reflective/rl CKPT
MODE=rl                             # zero_shot, cold_start, reflective, rl

echo "Processing shard $split of $all"


for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}
    IMAGE_FOLDER=$DATADIR
    echo "${QUESTION_FILE}--${IMAGE_FOLDER}"

    if [ -z "$IMAGE_FOLDER" ]; then
        echo "Warning: No image folder defined for $QUESTION_FILE. Skipping..."
        continue
    fi
    RESULTDIR=./eval/results/$MODE/${script_paths[i]}/ 
    mkdir -p $RESULTDIR

    # max-pixels: 448*28*28->351232
    python eval/infer.py \
        --model-path $CKPT \
        --model-name ${model_name} \
        --dataset ${script_paths[i]} \
        --input-file $DATADIR/${QUESTION_FILE}.json \
        --image-folder $IMAGE_FOLDER \
        --output-dir $RESULTDIR \
        --temperature 0.75 \
        --max-frames 16 \
        --max-pixels 351232 \
        --split $split \
        --all $all
done