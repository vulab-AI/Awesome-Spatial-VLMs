#!/bin/bash
pip install transformers==4.51.1 datasets==3.5.0 scipy einops sentencepiece protobuf uvicorn fastapi sse-starlette matplotlib fire pydantic==1.10.0 tensorboard==2.14.0
pip install protobuf==3.19.0 deepspeed==0.17.1 accelerate==1.6.0 nltk==3.9.1
pip install loguru omegaconf==2.3.0 tabulate

# vLLM support 
pip install vllm==0.7.3
pip install ray==2.47.1
pip install qwen-vl-utils
pip install codetiming==1.4.0 mathruler==0.1.0 pylatexenc==2.10 torchdata==0.11.0 omegaconf==2.3.0 wandb==0.19.11                # EasyR1 Requirements