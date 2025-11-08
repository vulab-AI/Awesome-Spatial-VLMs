#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train train/cold_start/vilasr_full_qwen2.5_vl_7b_8gpu/config_cold_start.yaml
