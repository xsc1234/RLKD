#!/bin/bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve
    --gpu-memory-utilization 0.45
    --model Qwen2.5-Math-7B