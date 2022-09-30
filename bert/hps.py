import os
import shutil
import json
from datetime import datetime
from typing import Optional

import wandb
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
# from ray.tune.integration.wandb import WandbLoggerCallback
from transformers import Trainer
from pathlib import Path

from transformers.trainer_utils import BestRun

RAY_RESULTS_DIR = "./results/ray_results"
BEST_TUNED_MODELS_PATH = "./results/best_tuned_models"


def get_immediate_child_directory_with_sub_name(parent_dir: str, sub_name: str) -> Optional[str]:
    directory_contents = os.listdir(parent_dir)
    for item in directory_contents:
        item_path = parent_dir + "/" + item
        if not os.path.isdir(item_path):
            continue
        if sub_name in item:
            return item

    return None


def save_best_model(best_run: BestRun, run_name: str) -> None:
    # Create directories
    Path(BEST_TUNED_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    target_dir = BEST_TUNED_MODELS_PATH + "/" + run_name
    # Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Save model and hyperparameter configuration
    tune_dir = RAY_RESULTS_DIR + "/" + run_name

    # Find best run directory
    best_run_dir = get_immediate_child_directory_with_sub_name(tune_dir, best_run.run_id)
    if best_run_dir is None:
        print(f"Could not find best run directory run_id={best_run.run_id}")
        return

    best_run_dir = tune_dir + "/" + best_run_dir
    # Go down two checkpoint levels
    for i in range(2):
        checkpoint_dir = get_immediate_child_directory_with_sub_name(best_run_dir, "checkpoint")
        if best_run_dir is None:
            print(f"Could not find best run checkpoint directory inside: {best_run_dir}")
            return

        best_run_dir += "/" + checkpoint_dir

    target_dir_path = Path(target_dir)
    if target_dir_path.exists() and target_dir_path.is_dir():
        print("Target directory already exists, removing/overriding:", target_dir)
        shutil.rmtree(target_dir_path)

    shutil.copytree(src=best_run_dir, dst=target_dir)
    # Write hyperparameters as json pretty print
    with open(target_dir + "/hyperparameters.json", 'w') as f:
        json.dump(best_run.hyperparameters, f, indent=True)


def tune_config_ray(trial):
    tune_config = {
        "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5, 4e-5]),
        "per_device_train_batch_size": tune.grid_search([16, 32])
    }
    return tune_config


def hp_tune(trainer: Trainer, model_name_or_path):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")

    run_name = model_name_or_path.replace("/", "_")
    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "per_device_train_batch_size": "bs",
        },
        metric_columns=[
            "epoch", "eval_loss", "eval_rmse"
            # "epoch", "training_iteration", "rmse", "eval_loss"
        ])

    best_run = trainer.hyperparameter_search(
        hp_space=tune_config_ray,
        backend='ray',
        metric='eval_loss',
        mode='min',
        direction='minimize',
        n_trials=1,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        keep_checkpoints_num=1,
        checkpoint_score_attr="eval_loss",
        fail_fast=True,
        progress_reporter=reporter,
        local_dir=RAY_RESULTS_DIR,
        name=run_name,
        callbacks=[WandbLoggerCallback(project="SuperLim2",
                                       entity="joeyohman",
                                       group=run_name + "_" + dt_string,
                                       # api_key="0fc05c8f0ff7f9219378a081a69de35fc26c1011",
                                       # api_key_file="wandb_key_file.txt",
                                       # log_config=data_args.tune_alg == 'PBT',
                                       # start_method="fork",
                                       # settings=wandb.Settings(start_method="fork"),
                                       )],
    )

    save_best_model(best_run, run_name)
    best_model_dir = BEST_TUNED_MODELS_PATH + "/" + run_name
    return best_run, best_model_dir
