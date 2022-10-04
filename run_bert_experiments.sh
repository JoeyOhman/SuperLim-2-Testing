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

run_cmd="python3 bert/bert_experiment_driver.py"

echo $run_cmd
$run_cmd


echo "Clearing ray HPO checkpoints!"
./clear_ray_results.sh
