#!/bin/bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve
    --gpu-memory-utilization 0.45 
    --model DeepSeek-R1-Distill-Qwen-7B