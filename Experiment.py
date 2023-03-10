import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np

from compute_metrics import metric_to_compute_fun
from dataset_loaders.dataset_loader import load_dataset_by_task, convert_label_to_original
from paths import get_experiment_metrics_path, get_experiment_predictions_path
from pathlib import Path

# Direction (min or max) is whether the optimization should minimize or maximize the metric
task_to_info_dict = {
    "ABSAbank-Imm": {"num_classes": 1, "metric": "krippendorff_interval", "direction": "max", "is_regression": True},
    "SweParaphrase": {"num_classes": 1, "metric": "krippendorff_interval", "direction": "max", "is_regression": True},
    "DaLAJ": {"num_classes": 2, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    "SweFAQ": {"num_classes": 2, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    "SweWiC": {"num_classes": 2, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    "SweWinograd": {"num_classes": 2, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    "SweMNLI": {"num_classes": 3, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    "ArgumentationSentences": {"num_classes": 3, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
    # "SweParaphrase": {"num_classes": 1, "metric": "rmse", "direction": "min", "is_regression": True},
    # "SweParaphrase": {"num_classes": 1, "metric": "spearmanr", "direction": "max"},
    # "DaLAJ": {"num_classes": 2, "metric": "accuracy", "direction": "max", "is_regression": False},
    # "SweFAQ": {"num_classes": 2, "metric": "accuracy", "direction": "max", "is_regression": False},
    # "ABSAbank-Imm": {"num_classes": 1, "metric": "rmse", "direction": "min", "is_regression": True},
    # "Reviews": {"num_classes": 2, "metric": "krippendorff_nominal", "direction": "max", "is_regression": False},
}


class Experiment(ABC):
    def __init__(self, task_name: str, model_name: str, data_fraction: float, evaluate_only: bool):
        self.task_name = task_name
        self.model_name = model_name
        self.data_fraction = data_fraction

        info_dict = task_to_info_dict[task_name]
        self.num_classes = info_dict["num_classes"]
        self.metric = info_dict["metric"]
        self.direction = info_dict["direction"]
        self.is_regression = info_dict["is_regression"]

        self.compute_metrics_fun = metric_to_compute_fun[self.metric]

        self.evaluate_only = evaluate_only

        experiment_metric_path = get_experiment_metrics_path(self.task_name, self.model_name)
        current_experiment_metric_path = experiment_metric_path + "/metrics.json"
        my_file = Path(current_experiment_metric_path)
        if my_file.is_file() and not evaluate_only:
            print(f"Metrics for model={self.model_name} and task={self.task_name} already exists, skipping!")
            exit()

    @abstractmethod
    def run_impl(self) -> Tuple[Dict[str, float], list, list]:
        pass

    def _get_jsonl_with_predictions(self, ds, predictions):
        samples = []
        for sample, prediction in zip(ds, predictions):
            # if np.ndim(predictions) > 1 and level_of_measurement != "interval":
            if (isinstance(prediction, list) or isinstance(prediction, np.ndarray)) and len(prediction) > 1:
                prediction = int(np.argmax(prediction))
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            if isinstance(prediction, list):
                assert len(prediction) == 1
                prediction = prediction[0]
            if isinstance(prediction, np.generic):
                prediction = prediction.item()
            line_dict = {k: v for k, v in sample.items()}
            prediction_org_format = convert_label_to_original(self.task_name, prediction)
            line_dict.update({"prediction": prediction_org_format})
            samples.append(line_dict)

        return samples

    def _save_predictions(self, predictions_eval, predictions_test):
        experiment_predictions_path = get_experiment_predictions_path(self.task_name, self.model_name)
        Path(experiment_predictions_path).mkdir(parents=True, exist_ok=True)

        _, dev_ds, test_ds = self._load_data_raw()
        for dev_test_name, predictions, ds in zip(
                ["/dev.jsonl", "/test.jsonl"], [predictions_eval, predictions_test], [dev_ds, test_ds]):

            predictions_list = self._get_jsonl_with_predictions(ds, predictions)
            # print(predictions_list[-1])
            strings_to_write = "\n".join([json.dumps(s, ensure_ascii=False) for s in predictions_list])

            with open(experiment_predictions_path + dev_test_name, 'w') as f:
                f.write(strings_to_write)

    def run(self) -> None:
        metric_dict: Dict[str, Any] = {
            "task": self.task_name,
            "model": self.model_name,
            "metric": self.metric,
            "direction": self.direction
        }
        results_metric_dict, predictions_eval, predictions_test = self.run_impl()
        metric_dict.update(results_metric_dict)

        experiment_metric_path = get_experiment_metrics_path(self.task_name, self.model_name)
        Path(experiment_metric_path).mkdir(parents=True, exist_ok=True)

        metric_str = json.dumps(metric_dict, ensure_ascii=False, indent="\t")
        print(metric_str)
        with open(experiment_metric_path + "/metrics.json", 'w') as f:
            f.write(metric_str)

        self._save_predictions(predictions_eval, predictions_test)

    def _load_data(self):
        return load_dataset_by_task(self.task_name, self.data_fraction)

    def _load_data_raw(self):
        return load_dataset_by_task(self.task_name, self.data_fraction, reformat=False)

    def _compute_metrics(self, predictions_labels_tuple):
        return self.compute_metrics_fun(predictions_labels_tuple)
