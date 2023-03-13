#!/bin/bash

# DEBUG will print directly to console instead of log files, and skip using wandb
DEBUG=1

# EVALUATE_ONLY will load existing metrics json and fine-tuned model, and overwrite the metrics json and predictions
EVALUATE_ONLY=0


# Only use wandb if not debugging
if [ "$DEBUG" == 0 ]; then
  # EDIT THIS TO YOUR OWN WANDB DETAILS & AVAILABLE GPUs
  export WANDB_ENTITY=joeyohman
  export WANDB_PROJECT=SuperLim2023
  api_key=$(cat api_wandb_key.txt)
  export WANDB_API_KEY=${api_key}
  # export CUDA_VISIBLE_DEVICES="4,5,6,7"
  export CUDA_VISIBLE_DEVICES="0,1"
else
  export WANDB_MODE=offline
  export WANDB_DISABLED=true
  # export CUDA_VISIBLE_DEVICES="7"
fi

# which python
# /bin/hostname -s
export PYTHONPATH="${pwd}:$PYTHONPATH"

mkdir -p logs

# Includes gpt models:
# declare -a models=("KB/bert-base-swedish-cased" "KBLab/megatron-bert-base-swedish-cased-600k" "KBLab/bert-base-swedish-cased-new" "xlm-roberta-base" "NbAiLab/nb-bert-base" "AI-Nordics/bert-large-swedish-cased" "KBLab/megatron-bert-large-swedish-cased-165k" "xlm-roberta-large" "gpt2" "AI-Sweden-Models/gpt-sw3-126m" "gpt2-medium" "AI-Sweden-Models/gpt-sw3-356m")

# All but gpt models:
declare -a models=("KB/bert-base-swedish-cased" "KBLab/megatron-bert-base-swedish-cased-600k" "KBLab/bert-base-swedish-cased-new" "xlm-roberta-base" "NbAiLab/nb-bert-base" "AI-Nordics/bert-large-swedish-cased" "KBLab/megatron-bert-large-swedish-cased-165k" "xlm-roberta-large")

# All Tasks
declare -a tasks=("ArgumentationSentences" "ABSAbank-Imm" "SweParaphrase" "SweFAQ" "SweWiC" "DaLAJ" "SweWinograd" "SweMNLI")
# declare -a tasks=("ABSAbank-Imm" "SweParaphrase" "SweWiC" "DaLAJ" "SweWinograd" "SweMNLI")

# Overwrite for single model or task
# declare -a models=("KB/bert-base-swedish-cased")
# declare -a tasks=("ABSAbank-Imm")

# Loop through models
for model in "${models[@]}"
do
    # Inner loop through tasks
    for task in "${tasks[@]}"
    do
        # Replace slash with dash in model name to avoid creating sub-directories
        safe_model_name="${model/"/"/-}"
        metrics_file="results/experiments/metrics/${task}/${safe_model_name}/metrics.json"
        if [ "$EVALUATE_ONLY" == 0 ]; then
          if test -f "$metrics_file"; then
              echo "$metrics_file exists, SKIPPING."
              continue
          fi
        fi
        log_file_path="logs/log_$(date +"%Y-%m-%d_%H:%M:%S")_${safe_model_name}_${task}.txt"
        run_cmd="python3 bert/bert_experiment_driver.py --model_name $model --task_name $task"
        if [ "$EVALUATE_ONLY" == 1 ]; then
          run_cmd="${run_cmd} --evaluate_only"
        fi
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
