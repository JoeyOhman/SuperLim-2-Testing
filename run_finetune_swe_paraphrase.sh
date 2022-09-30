#!/bin/bash

export WANDB_ENTITY=joeyohman
export WANDB_PROJECT=SuperLim2
api_key=$(cat api_wandb_key.txt)
export WANDB_API_KEY=${api_key}
# export WANDB_MODE=offline
# export WANDB_DISABLED=true

# which python
# /bin/hostname -s
export PYTHONPATH="${pwd}:$PYTHONPATH"

MODEL_NAME="KB/bert-base-swedish-cased"
# MODEL_NAME="microsoft/Multilingual-MiniLM-L12-H384"
# MODEL_NAME="KBLab/megatron-bert-base-swedish-cased-600k"
# MODEL_NAME="KBLab/megatron-bert-large-swedish-cased-165k"
# --eval_steps 200 \

run_cmd="python3 bert/finetune_swe_paraphrase.py
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ./results \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 10 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --skip_memory_metrics \
        --fp16 \
        --disable_tqdm 1 \
        --weight_decay 0.1 \
        --learning_rate 2e-5 \
        --max_input_length 128 \
        --warmup_ratio 0.06 \
        --load_best_model_at_end 1 \
        --data_fraction 1.0 \
        --hp_search 1 \
        --report_to none
        "

echo $run_cmd
$run_cmd

echo "Clearing ray HPO checkpoints!"
./clear_ray_results.sh
