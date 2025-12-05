### Source Code for paper "RLKD: Distilling LLMs’ Reasoning via Reinforcement Learning"

#### Install
```
pip install -r requirements.txt
```

#### Training:
##### Data Construction for Training Generative Structure Reward Model
```
python ./make_data/make_data.py \
    --dataset_path /your/path/original_dataset.json \
    --random_sampled_path /your/path/sampled_data.joblib \
    --save_path /your/path/processed_results.json

python ./open-r1-main/src/open_r1/make_training_dataset_GSRM.py \
--input_path /your/path/processed_results.json \
--output_path /your/output/GSRM_training.json
```

##### Train Generative Structure Reward Model
```
CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info accelerate launch
    --main_process_port 29536 
    --num_processes 2 
    --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft-GSRM.py 
    --config recipes/GSRM-7B/sft/SFT_GSRM.yaml
    --data_path /your/output/GSRM_training.json #your constructed GSRM sft json path
```

##### Data Construction for RLKD
```
python ./open-r1-main/src/open_r1/make_training_dataset_RLKD.py \
    --input_path /path/to/Open-R1-math.json \
    --train_output /path/to/train_output.json \
    --test_output /path/to/test_output.json
```

##### Train RLKD
```
nohup sh ./open-r1-main/src/open_r1/grpo_inference.sh > grpo_inference.log 2>&1 &
nohup sh ./open-r1-main/src/open_r1/grpo_GSRM_trl.sh > grpo_GSRM_trl.log 2>&1 &
nohup sh ./open-r1-main/src/open_r1/grpo_match.sh > grpo_match.log 2>&1 &
tmux new-session -d -s grpo_session "sh ./open-r1-main/src/open_r1/grpo_RLKD.sh > grpo_RLKD_openr1-math.log 2>&1"
```


##### Train RLKD-zero
```
nohup sh ./open-r1-main/src/open_r1/grpo_inference_zero.sh > grpo_inference_zero.log 2>&1 &
nohup sh ./open-r1-main/src/open_r1/grpo_GSRM_trl.sh > grpo_GSRM_trl.log 2>&1 &
nohup sh ./open-r1-main/src/open_r1/grpo_match.sh > grpo_match.log 2>&1 &
tmux new-session -d -s grpo_session "sh ./open-r1-main/src/open_r1/grpo_RLKD_zero.sh > grpo_RLKD_zero_openr1-math.log 2>&1"
```

#### Evaluation

need lighteval==0.9.0
##### AIME
```
NUM_GPUS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
MODEL=./Your_Model_PATH
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=./Your_Output_PATH/$MODEL
export CUDA_VISIBLE_DEVICES=6
nohup lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" --use-chat-template --output-dir $OUTPUT_DIR > aime.log 2>&1 &
```

##### GPQA
```
NUM_GPUS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
MODEL=./Your_Model_PATH
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=./Your_Output_PATH/$MODEL
export CUDA_VISIBLE_DEVICES=6
nohup lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" --use-chat-template --output-dir $OUTPUT_DIR > gpqa.log 2>&1 &
```

##### MATH-500
```
NUM_GPUS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
MODEL=./Your_Model_PATH
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=./Your_Output_PATH/$MODEL
export CUDA_VISIBLE_DEVICES=6
nohup lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" --use-chat-template --output-dir $OUTPUT_DIR > math_500.log 2>&1 &

```
