import json
from abc import ABC, abstractmethod
from typing import Dict, Any

from compute_metrics import metric_to_compute_fun
from dataset_loaders.dataset_loader import load_dataset_by_task
from paths import get_experiment_metrics_path
from pathlib import Path

# Direction (min or max) is whether the optimization should minimize or maximize the metric
task_to_info_dict = {
    "SweParaphrase": {"num_classes": 1, "metric": "rmse", "direction": "min"},
    "DaLAJ": {"num_classes": 2, "metric": "accuracy", "direction": "max"},
    "SweFAQ": {"num_classes": 2, "metric": "accuracy", "direction": "max"},
}


class Experiment(ABC):
    def __init__(self, task_name: str, model_name: str, data_fraction: float):
        self.task_name = task_name
        self.model_name = model_name
        self.data_fraction = data_fraction

        info_dict = task_to_info_dict[task_name]
        self.num_classes = info_dict["num_classes"]
        self.metric = info_dict["metric"]
        self.direction = info_dict["direction"]

        self.compute_metrics_fun = metric_to_compute_fun[self.metric]

    @abstractmethod
    def run_impl(self) -> Dict[str, float]:
        pass

    def run(self) -> None:
        metric_dict: Dict[str, Any] = {
            "task": self.task_name,
            "model": self.model_name
        }
        results_metric_dict = self.run_impl()
        metric_dict.update(results_metric_dict)
        # experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=self.model_name, task=self.task_name)
        experiment_metric_path = get_experiment_metrics_path(self.task_name,  self.model_name)
        Path(experiment_metric_path).mkdir(parents=True, exist_ok=True)
        with open(experiment_metric_path + "/metrics.json", 'w') as f:
            f.write(json.dumps(metric_dict, ensure_ascii=False, indent="\t"))

    def _load_data(self):
        return load_dataset_by_task(self.task_name, self.data_fraction)

    def _compute_metrics(self, predictions_labels_tuple):
        return self.compute_metrics_fun(predictions_labels_tuple)
