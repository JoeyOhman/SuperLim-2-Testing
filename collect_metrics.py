import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

from paths import METRICS_PATH, RESULTS_PATH


def load_all_metric_dicts():
    metric_dicts = []
    for path in Path(METRICS_PATH).rglob('metrics.json'):
    # for path in Path(RESULTS_PATH + '/downloaded_metrics/').rglob('*.json'):
        with open(path, 'r') as f:
            json_dict = json.load(f)

        metric_dicts.append(json_dict)

    return metric_dicts


def combine_dicts(metric_dicts):
    # task_to_models = defaultdict(list)
    task_and_model_to_eval = defaultdict(dict)
    task_to_metric_name_and_direction = {}
    for metric_dict in metric_dicts:
        task = metric_dict["task"]
        model = metric_dict["model"]
        direction = metric_dict["direction"]
        metric_name = metric_dict["metric"]
        eval = metric_dict["eval"]
        test = metric_dict["test"]
        # task_to_models[task][model] = eval
        # task_to_models[task].append((model, eval))
        task_and_model_to_eval[task][model] = round(eval, 3)
        task_to_metric_name_and_direction[task] = metric_name, direction
        # task_to_models[task].append((model, test))

    # print(task_to_models)
    # for task, model_list in task_to_models.items():
    #     print("*" * 50)
    #     print(task)
    #     for model, metric in model_list:
    #         print(model, metric)

    return task_and_model_to_eval, task_to_metric_name_and_direction


def print_tables(task_and_model_to_eval_dict, task_to_metric_name_and_direction_dict):
    # task_to_metric = {
    #     "SweParaphrase": "rmse",
    #     "DaLAJ": "accuracy",
    #     "SweFAQ": "accuracy",
    #     "ABSAbank-Imm": "rmse"
    # }

    tasks = sorted(task_to_metric_name_and_direction_dict.keys())
    task_headers = [t + " (" + task_to_metric_name_and_direction_dict[t][0].replace("accuracy", "acc") + ")"
                    for t in tasks]
    headers = ["Model"] + task_headers + ["Sum (+acc, -rmse)"]
    sorted_model_names = []
    # print(headers)
    task_columns = []
    for task in tasks:
        # print("*" * 50)
        # print(task)
        # metric_name, direction = task_to_metric_name_and_direction_dict[task]
        model_to_eval_dict = task_and_model_to_eval_dict[task]
        sorted_models_to_metric_list = sorted(model_to_eval_dict.items(), key=lambda x: x[0])
        sorted_model_names = [model_name for model_name, metric in sorted_models_to_metric_list]
        task_metrics = [metric for model, metric in sorted_models_to_metric_list]
        task_columns.append(task_metrics)

    complete_rows = []
    model_rows = list(map(list, zip(*task_columns)))
    for model_name, row in zip(sorted_model_names, model_rows):
        agg = -row[0] + row[1] + row[2] - row[3]
        row = [model_name] + row + [agg]
        complete_rows.append(row)

    complete_rows.sort(key=lambda x: x[-1], reverse=True)
    table = tabulate(complete_rows, headers=headers, tablefmt="github")
    print(table)
    return

    for task, model_list in task_to_models_dict.items():
        direction = task_to_direction_dict[task]
        model_list.sort(key=lambda x: x[-1], reverse=direction == "max")

        table = tabulate(model_list, headers=["model", task_to_metric[task]], tablefmt="github")
        print("*" * 50)
        print(task)
        print(table)


def main():
    metric_dicts = load_all_metric_dicts()
    # print(len(metric_dicts))
    # print(metric_dicts[0])
    task_and_model_to_eval_dict, task_to_metric_name_and_direction = combine_dicts(metric_dicts)
    print_tables(task_and_model_to_eval_dict, task_to_metric_name_and_direction)


if __name__ == '__main__':
    main()
