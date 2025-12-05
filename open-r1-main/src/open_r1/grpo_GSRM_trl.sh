#!/bin/bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve
    --port 29533
    --gpu-memory-utilization 0.5
    --model ./Your_GSRM_PATH