import os
import shutil
import json
from datetime import datetime
from typing import Optional

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from transformers import Trainer
from pathlib import Path

from transformers.trainer_utils import BestRun

from paths import RAY_RESULTS_PATH, get_experiment_metrics_path, get_experiment_models_path


def get_immediate_child_directory_with_sub_name(parent_dir: str, sub_name: str) -> Optional[str]:
    directory_contents = os.listdir(parent_dir)
    for item in directory_contents:
        item_path = parent_dir + "/" + item
        if not os.path.isdir(item_path):
            continue
        if sub_name in item:
            return item

    return None


def save_best_model(best_run: BestRun, run_name: str, model_dir: str, metric_dir) -> None:
    # Create directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(metric_dir).mkdir(parents=True, exist_ok=True)
    # target_dir = BEST_TUNED_MODELS_PATH + "/" + run_name
    # Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Save model and hyperparameter configuration
    tune_dir = RAY_RESULTS_PATH + "/" + run_name

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

    target_dir_path = Path(model_dir)
    if target_dir_path.exists() and target_dir_path.is_dir():
        print("Target directory already exists, removing/overriding:", model_dir)
        shutil.rmtree(target_dir_path)

    shutil.copytree(src=best_run_dir, dst=model_dir)
    # Write hyperparameters as json pretty print
    with open(metric_dir + "/hyperparameters.json", 'w') as f:
        json.dump(best_run.hyperparameters, f, indent=True)


def tune_config_ray(trial):
    tune_config = {
        "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5, 4e-5]),
        "per_device_train_batch_size": tune.grid_search([16, 32])
    }
    return tune_config


def tune_config_ray_quick(trial):
    tune_config = {
        "learning_rate": tune.grid_search([1e-5, 3e-5]),
        "per_device_train_batch_size": tune.grid_search([16])
    }
    return tune_config


def hp_tune(trainer: Trainer, model_name_or_path: str, task_name: str, quick_run: bool,
            metric: Optional[str] = None, direction: str = "min"):
    assert direction in ["min", "max"]
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")

    # experiment_model_path = EXPERIMENT_MODELS_PATH_TEMPLATE.format(model=model_name_or_path, task=task_name)
    # experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=model_name_or_path, task=task_name)

    experiment_model_path = get_experiment_models_path(task_name, model_name_or_path)
    experiment_metric_path = get_experiment_metrics_path(task_name, model_name_or_path)

    tune_config = tune_config_ray_quick if quick_run else tune_config_ray

    # run_name = model_name_or_path.replace("/", "_")
    run_name = task_name + "_" + model_name_or_path.replace("/", "-")
    metrics_to_report = ["epoch", "eval_loss"]
    if metric is None:
        metric = "eval_loss"
        direction = "min"
    else:
        metric = "eval_" + metric
        metrics_to_report.append(metric)
    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "per_device_train_batch_size": "bs",
        },
        metric_columns=metrics_to_report
        # metric_columns=[
        #     "epoch", "eval_loss", "eval_rmse"
        # ]
        )

    best_run = trainer.hyperparameter_search(
        hp_space=tune_config,
        backend='ray',
        metric=metric,
        # mode='min',
        # direction='minimize'
        mode=direction,
        direction=direction + "imize",
        n_trials=1,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        keep_checkpoints_num=1,
        checkpoint_score_attr=metric,
        fail_fast=True,
        progress_reporter=reporter,
        local_dir=RAY_RESULTS_PATH,
        name=run_name,
        callbacks=[WandbLoggerCallback(project="SuperLim2",
                                       entity="joeyohman",
                                       group=run_name + "_" + dt_string,
                                       # settings=wandb.Settings(start_method="fork"),
                                       )],
    )

    save_best_model(best_run, run_name, experiment_model_path, experiment_metric_path)
    # best_model_dir = BEST_TUNED_MODELS_PATH + "/" + run_name + "/model"
    return best_run, experiment_model_path
