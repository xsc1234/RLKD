#!/bin/bash
CUDA_VISIBLE_DEVICES=7 vllm serve
    --port 29564
    --dtype float16
    --gpu-memory-utilization 1.0
    --model Qwen2.5-7B-Instruct