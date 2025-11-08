#!/bin/bash
set -x

MODEL_PATH=checkpoints/reflective/qwen2_5_vl-7b_vilasr_reflective              # checkpoint after reflective rejection sampling

# The system_prompt defined here has no effect; it has already been implemented directly in the code.
SYSTEM_PROMPT="""\n### Guidance:\nYou are a spatial reasoning assistant with access to two powerful visualization tools.\nYour task is to break down complex spatial problems and iteratively refine your solution through visualization feedback.\n\nAvailable tools:\nYou can use below two tools to visualize, and you should wait the visualization feedback for refine your reasoning.\n\n1. **Object Mapper**\n- Purpose: Identifies and maps key items in the space\n- Input format: JSON\n```json\n[{\n    "bbox_2d": [x1, y1, x2, y2],\n    "label": "object name/description"\n}]\n```\n- Output: Generates bounding boxes for visual inspection\n\n2. **Path Tracer**\n- Purpose: Plots movement or connections between points\n- Input format: JSON\n```json\n[{\n    "start_point_2d": [x1, y1],\n    "end_point_2d": [x2, y2],\n    "label": "trace_description"\n}]\n```\n- Output: Generates visual paths for verification\n\n### Required Output Format:\nFor each reasoning step, you must structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: [Object Mapper/Path Tracer]\n```json\n[JSON format coordinates]\n```\n\nAfter your reasoning and iteratively refine your solution through visualization feedback, you should arrive at a final answer and structure your response as follows:\n<think> [Your detailed reasoning process] </think> Action: Answer\n<answer> [Your final answer] </answer>\n\n### Solution Process:\n1. Initial Analysis\n   - Break down the spatial problem\n   - Plan your approach\n\n2. Iterative Reasoning\n   For each step:\n   - Choose appropriate tool\n   - Provide coordinates in JSON format\n   - Observe the visualization output\n   - Reflect on the results:\n     * Is the placement/path accurate?\n     * Does it align with your reasoning?\n     * What adjustments are needed?\n   - Refine coordinates if necessary\n\n3. Validation\n   - Review complete solution\n   - Ensure all steps logically connect\n   - Verify final answer matches visual evidence\n\n4. Backtrack and Adjust\n   - If errors are found, backtrack to previous steps\n   - Modify actions or decisions as needed\n   - Re-execute subsequent steps to validate corrections\n\n5. Summary and Ouput\n   - Summarize the entire process and give final answer"""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m verl.trainer.main \
    config=scripts/config_grpo.yaml \
    data.train_files=./ViLaSR-data/rl/vilasr_rl_data.jsonl \
    data.val_files=./ViLaSR-data/rl/vilasr_rl_data.jsonl \
    data.max_response_length=13312 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    algorithm.kl_coef=0.0 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.n=8 \
    worker.rollout.num_llm_calls_available=10 \
    worker.rollout.limit_images=45 \
    trainer.experiment_name=stage3_v6_continue400 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path='checkpoints/rl/qwen2_5_vl-7b_vilasr_grpo' 