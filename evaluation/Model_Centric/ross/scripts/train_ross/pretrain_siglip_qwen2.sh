#!/bin/bash

set -x

torchrun --nproc-per-node=8 --nnodes $1 --node_rank $2 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --mm_inv_projector_lr 1e-4 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --output_dir ./checkpoints/ross-siglip-qwen2-7b-pt558k \
    --vision_tower /data/llm_model_zoo/siglip-so400m-patch14-384 \
    --version plain \
    --mm_pixel_decoder /data/llm_model_zoo/FLUX.1-dev \
    \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "ross-siglip-qwen2-7b-pt558k"
