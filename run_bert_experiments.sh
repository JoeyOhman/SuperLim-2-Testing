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

pip install evaluate

mkdir -p logs
log_file_path="logs/log_$(date +"%Y-%m-%d_%H:%M:%S").txt"

run_cmd="python3 bert/bert_experiment_driver.py"

echo $run_cmd
echo "Writing output to: ${log_file_path}"
$run_cmd &> ${log_file_path}

./clear_ray_results.sh
