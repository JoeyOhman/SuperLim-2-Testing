import json
import os
import shutil
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from paths import TRAINER_OUTPUT_PATH, get_experiment_models_path
from utils import get_device


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_config(model_name_or_path, num_labels):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    return config


def load_model(model_name_or_path, config):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.to(get_device())
    return model


def _move_best_model_and_clean(best_run, best_model_path, task_name, model_name):
    experiment_model_path = get_experiment_models_path(task_name, model_name)
    experiment_task_path = "/".join(experiment_model_path.split("/")[:-1])
    Path(experiment_task_path).mkdir(parents=True, exist_ok=True)

    experiment_model_path_obj = Path(experiment_model_path)
    if experiment_model_path_obj.exists() and experiment_model_path_obj.is_dir():
        print("Target directory already exists, removing/overriding:", experiment_model_path)
        shutil.rmtree(experiment_model_path_obj)
    shutil.copytree(src=best_model_path, dst=experiment_model_path)
    # os.remove(experiment_model_path + "/trainer_state.json")
    new_path_best_model = experiment_model_path

    hps_session_id = best_run.run_id.split("_")[0]
    session_models_prefix = "run-" + hps_session_id
    for path in Path(TRAINER_OUTPUT_PATH).rglob(session_models_prefix + "*"):
        # print(path)
        shutil.rmtree(path)

    return new_path_best_model


def find_best_model_and_clean(best_run, metric_direction, task_name, model_name):
    hps_session_id = best_run.run_id.split("_")[0]
    session_models_prefix = "run-" + hps_session_id

    path_best_metric_tuples = []
    for path in Path(TRAINER_OUTPUT_PATH).rglob(session_models_prefix + "*"):
        # Find the latest checkpoint, that contains info from all epochs/checkpoints
        tups = [(cp_path, int(str(cp_path).split("-")[-1])) for cp_path in path.rglob("checkpoint-*")]
        max_checkpoint_path = max(tups, key=lambda x: x[1])[0]

        # Extract the best checkpoint path and its metric
        with open(str(max_checkpoint_path) + "/trainer_state.json", 'r') as f:
            trainer_state = json.load(f)
        best_checkpoint_in_trial_path = trainer_state["best_model_checkpoint"]
        best_checkpoint_in_trial_metric = trainer_state["best_metric"]

        # Add it as tuple to list of checkpoints
        tup = (best_checkpoint_in_trial_path, best_checkpoint_in_trial_metric)
        path_best_metric_tuples.append(tup)

    # Select best one according to direction of metric
    if metric_direction == "min":
        best_model_path = min(path_best_metric_tuples, key=lambda x: x[1])[0]
    else:
        best_model_path = max(path_best_metric_tuples, key=lambda x: x[1])[0]

    new_path_best_model = _move_best_model_and_clean(best_run, best_model_path, task_name, model_name)

    return new_path_best_model
