#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
JSON_FOLDER="./playground/data/annotations"
VIDEO_FOLDER="./playground/data/LLaVA-3D-Pretrain"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-3d-7b-pretrain \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava-3d-pretrain.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower SpatialAwareModule \
    --num_frames 20 \
    --num_sample_tokens 1152 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-3d-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
