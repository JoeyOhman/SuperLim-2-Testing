#!/bin/bash

export WANDB_MODE=offline
# export WANDB_DISABLED=true

# which python
# /bin/hostname -s
export PYTHONPATH="${pwd}:$PYTHONPATH"

MODEL_NAME="KB/bert-base-swedish-cased"
# MODEL_NAME="KBLab/megatron-bert-base-swedish-cased-600k"
# MODEL_NAME="KBLab/megatron-bert-large-swedish-cased-165k"

run_cmd="python bert/finetune_swe_paraphrase.py
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ./results \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 5 \
        --evaluation_strategy steps \
        --save_strategy epoch \
        --skip_memory_metrics \
        --eval_steps 100 \
        --fp16 \
        --disable_tqdm 1 \
        --weight_decay 0.01 \
        --learning_rate 2e-5 \
        --max_input_length 128 \
        --warmup_ratio 0.05 \
        --data_fraction 1.0
        "

echo $run_cmd
$run_cmd