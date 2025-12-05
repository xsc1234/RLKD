#!/bin/bash
CUDA_VISIBLE_DEVICES=1,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch 
    --config_file recipes/accelerate_configs/zero3.yaml 
    --num_processes 4 src/open_r1/grpo_RLKD.py 
    --config recipes/RLKD-based-on-DeepSeek-R1-Distill-Qwen-7B/config_demo_openr1-math.yaml