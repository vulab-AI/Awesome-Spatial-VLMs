#!/bin/bash

FORCE_TORCHRUN=1 \
NNODES=4 \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=29500 \
llamafactory-cli train train/cold_start/vilasr_full_qwen2.5_vl_7b_32gpu/config_cold_start.yaml
