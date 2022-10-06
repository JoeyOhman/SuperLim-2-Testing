import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

from paths import METRICS_PATH, RESULTS_PATH


def load_all_metric_dicts():
    metric_dicts = []
    # for path in Path(METRICS_PATH).rglob('metrics.json'):
    for path in Path(RESULTS_PATH + '/downloaded_metrics/').rglob('*.json'):
        print(path.name)
        with open(path, 'r') as f:
            json_dict = json.load(f)

        metric_dicts.append(json_dict)

    return metric_dicts


def combine_dicts(metric_dicts):
    task_to_models = defaultdict(list)
    for metric_dict in metric_dicts:
        task = metric_dict["task"]
        model = metric_dict["model"]
        eval = metric_dict["eval"]
        test = metric_dict["test"]
        # task_to_models[task][model] = eval
        task_to_models[task].append((model, eval))
        # task_to_models[task].append((model, test))

    print(task_to_models)
    for task, model_list in task_to_models.items():
        print("*" * 50)
        print(task)
        for model, metric in model_list:
            print(model, metric)

    return task_to_models


def print_tables(task_to_models_dict):
    task_to_metric = {
        "SweParaphrase": "rmse",
        "DaLAJ": "accuracy",
        "SweFAQ": "accuracy"
    }

    for task, model_list in task_to_models_dict.items():
        model_list.sort(key=lambda x: x[1], reverse=task != "SweParaphrase")

        table = tabulate(model_list, headers=["model", task_to_metric[task]], tablefmt="github")
        print("*" * 50)
        print(task)
        print(table)


def main():
    metric_dicts = load_all_metric_dicts()
    print(len(metric_dicts))
    print(metric_dicts[0])
    task_to_models_dict = combine_dicts(metric_dicts)
    print_tables(task_to_models_dict)


if __name__ == '__main__':
    main()
