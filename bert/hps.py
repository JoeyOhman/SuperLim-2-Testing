import os
import shutil
import json
from functools import partial
from datetime import datetime
from typing import Optional, List, Dict

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import Callback
from ray.tune import CLIReporter
from transformers import Trainer
from pathlib import Path

from transformers.trainer_utils import BestRun

from paths import RAY_RESULTS_PATH, get_experiment_metrics_path, get_experiment_models_path

"""
class ManualMetricsCallback(Callback):

    def __init__(self, metric):
        self.metric = metric
        self.run_to_metric_history = {}

    def on_trial_result(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        result: Dict,
        **info,
    ):
        # self.run_to_metric_history[]
        # print("*" * 50)
        # print("iteration:", iteration)
        # print("result:", result)
        # print("trial_id:", trial.trial_id)
        # print("config:", trial.config)
        # print("evaluated_params", trial.evaluated_params)
        # print("best_result:", trial.best_result)
        # print("metric_analysis:", trial.metric_analysis)
        # print("param_config:", trial.param_config)
        # print("trial.results:", trial.results)

        trial_id = result["trial_id"]
        epoch = result["epoch"]
        eval_metric = result[self.metric]
        eval_loss = result["eval_loss"]
        hyperparams = result["config"]
        # self.run_to_metric_history[]


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
"""


def tune_config_ray(accumulation_steps, trial):
    tune_config = {
        "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5, 4e-5]),
        "per_device_train_batch_size": tune.grid_search([
            int(16 / accumulation_steps),
            int(32 / accumulation_steps)
        ])
    }
    return tune_config


def tune_config_ray_quick(accumulation_steps, trial):
    tune_config = {
        "learning_rate": tune.grid_search([1e-5, 3e-5]),
        "per_device_train_batch_size": tune.grid_search([int(16 / accumulation_steps)])
    }
    return tune_config


def hp_tune(trainer: Trainer, model_name_or_path: str, task_name: str, accumulation_steps: int, quick_run: bool,
            metric: Optional[str] = None, direction: str = "min"):
    assert direction in ["min", "max"]
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")

    # experiment_model_path = get_experiment_models_path(task_name, model_name_or_path)
    # experiment_metric_path = get_experiment_metrics_path(task_name, model_name_or_path)

    tune_config = tune_config_ray_quick if quick_run else tune_config_ray
    tune_config = partial(tune_config, accumulation_steps)

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
        callbacks=[
            # ManualMetricsCallback(metric),
            WandbLoggerCallback(project="SuperLim2023",
                                entity="joeyohman",
                                group=run_name + "_" + dt_string,
                                # settings=wandb.Settings(start_method="fork"),
                                )
        ],
    )

    # save_best_model(best_run, run_name, experiment_model_path, experiment_metric_path)
    # best_model_dir = BEST_TUNED_MODELS_PATH + "/" + run_name + "/model"
    # return best_run, experiment_model_path
    return best_run
