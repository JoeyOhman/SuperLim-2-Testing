import os
import pathlib

ABSOLUTE_PATH_PROJ = str(pathlib.Path(__file__).parent.resolve())

# DATASET_PATH = os.path.join(ABSOLUTE_PATH_PROJ, "data", "_data_old")
DATASET_PATH = os.path.join(ABSOLUTE_PATH_PROJ, "data")
TSV_PATH = os.path.join(DATASET_PATH, "tsv")
JSONL_PATH = os.path.join(DATASET_PATH, "jsonl")

RESULTS_PATH = os.path.join(ABSOLUTE_PATH_PROJ, "results")
TRAINER_OUTPUT_PATH = os.path.join(RESULTS_PATH, "trainer_output")
RAY_RESULTS_PATH = os.path.join(RESULTS_PATH, "ray_results")

EXPERIMENTS_PATH = os.path.join(RESULTS_PATH, "experiments")
METRICS_PATH = os.path.join(EXPERIMENTS_PATH, "metrics")
MODELS_PATH = os.path.join(EXPERIMENTS_PATH, "models")
PREDICTIONS_PATH = os.path.join(EXPERIMENTS_PATH, "predictions")

EXPERIMENT_METRICS_PATH_TEMPLATE = os.path.join(METRICS_PATH, "{task}", "{model}")
EXPERIMENT_PREDICTIONS_PATH_TEMPLATE = os.path.join(PREDICTIONS_PATH, "{model}", "{task}")
EXPERIMENT_MODELS_PATH_TEMPLATE = os.path.join(MODELS_PATH, "{task}", "{model}")


def get_experiment_metrics_path(task: str, model: str) -> str:
    safe_task = task.replace("/", "-")
    safe_model = model.replace("/", "-")
    return EXPERIMENT_METRICS_PATH_TEMPLATE.format(task=safe_task, model=safe_model)


def get_experiment_predictions_path(task: str, model: str) -> str:
    safe_task = task.replace("/", "-")
    safe_model = model.replace("/", "-")
    return EXPERIMENT_PREDICTIONS_PATH_TEMPLATE.format(model=safe_model, task=safe_task)


def get_experiment_models_path(task: str, model: str) -> str:
    safe_task = task.replace("/", "-")
    safe_model = model.replace("/", "-")
    return EXPERIMENT_MODELS_PATH_TEMPLATE.format(task=safe_task, model=safe_model)
