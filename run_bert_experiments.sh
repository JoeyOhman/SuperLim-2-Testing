#!/bin/bash

DEBUG=0

# Only use wandb if not debugging
if [ "$DEBUG" == 0 ]; then
  export WANDB_ENTITY=joeyohman
  export WANDB_PROJECT=SuperLim2023
  api_key=$(cat api_wandb_key.txt)
  export WANDB_API_KEY=${api_key}
  export CUDA_VISIBLE_DEVICES="4,5,6,7"
else
  export WANDB_MODE=offline
  export WANDB_DISABLED=true
  export CUDA_VISIBLE_DEVICES="7"
fi

# which python
# /bin/hostname -s
export PYTHONPATH="${pwd}:$PYTHONPATH"

# These pip installs are setup required in the docker container, skip if debug (which is used locally)
# if [ "$DEBUG" == 0 ]; then
#   pip install evaluate
#   pip install sentencepiece
# fi

mkdir -p logs

# declare -a models=("KB/bert-base-swedish-cased" "KBLab/megatron-bert-base-swedish-cased-600k" "")
# declare -a models=("KB/bert-base-swedish-cased")
declare -a models=("KB/bert-base-swedish-cased" "KBLab/megatron-bert-base-swedish-cased-600k" "KBLab/bert-base-swedish-cased-new" "xlm-roberta-base" "gpt2" "AI-Sweden-Models/gpt-sw3-126m" "gpt2-medium" "AI-Sweden-Models/gpt-sw3-356m" "xlm-roberta-large")
# declare -a models=("gpt2")
# declare -a models=("gpt2-medium")
# declare -a models=("AI-Sweden/gpt-sw3-356m-private")
# declare -a models=("microsoft/mdeberta-v3-base")
# declare -a tasks=("ABSAbankImm" "DaLAJ" "SweFAQ" "SweParaphrase")
declare -a tasks=("ABSAbankImm" "SweParaphrase" "DaLAJ")
# declare -a tasks=("Reviews")

# Loop through models
for model in "${models[@]}"
do
    # Inner loop through tasks
    for task in "${tasks[@]}"
    do
        # Replace slash with dash in model name to avoid creating sub-directories
        safe_model_name="${model/"/"/-}"
        log_file_path="logs/log_$(date +"%Y-%m-%d_%H:%M:%S")_${safe_model_name}_${task}.txt"
        run_cmd="python3 bert/bert_experiment_driver.py --model_name $model --task_name $task"
        echo "****************************************************************************************"
        echo "Model=$model"
        echo "Task=$task"
        echo "Writing output to: $log_file_path"

        echo $run_cmd

        # Use terminal directly for debugging
        if [ "$DEBUG" == 1 ]; then
          $run_cmd
        else
          $run_cmd &> ${log_file_path}
        fi

        ./clear_result_checkpoints.sh
    done
done
exit

run_cmd="python3 bert/bert_experiment_driver.py"

echo $run_cmd
echo "Writing output to: ${log_file_path}"
$run_cmd &> ${log_file_path}
# $run_cmd

./clear_result_checkpoints.sh

echo "Compressing metrics directory"
tar czf metrics.tar.gz results/experiments/metrics

# echo "Compressing models directory"
# tar czf metrics.tar.gz results/experiments/models

echo "Done!"
