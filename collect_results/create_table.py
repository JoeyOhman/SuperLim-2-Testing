from collections import defaultdict

import numpy as np
from tabulate import tabulate

from collect_results.collect_model_metrics import create_and_load_model_dicts

TEST_SCORES = False
INCLUDE_GPT = False


def avg_distance_from_max(metrics):
    max_val = max(metrics)
    metrics_wo_max = [m for m in metrics if m != max_val]
    distances = [abs(max_val - m) for m in metrics_wo_max]
    if sum(distances) == 0:
        return 0
    return np.mean(distances)


def create_row(model_dict, task_triples):
    model_name = model_dict["model"]["model_name"]
    row = [model_name]
    for task, _, _ in task_triples:
        task_dict = model_dict["tasks"][task]
        if task == "SweWinogender":
            row.append(task_dict["test"]["parity"])
            row.append(task_dict["test"]["alpha"])
        else:
            row.append(task_dict["test" if TEST_SCORES else "dev"])

    if len(task_triples) > 1:
        scores = [col if task_triples[idx][-1] == "max" else -col for idx, col in enumerate(row[1:])]
        scores = [-100 if score is None else score for score in scores]
        agg_column = np.mean(scores)
        row.append(agg_column)
    return row


def create_table(task_triples, rows):
    headers = ["Model"]
    for task, metric_name, direction in task_triples:
        # print(task, metric_name, direction)
        # header = task + f" ({metric_name.replace('accuracy', 'acc')}) " + ("↑" if direction == "max" else "↓")
        header = task
        headers.append(header)
    headers.append("Avg ↑")

    rows_new = []
    for r in rows:
        if "gpt" in r[0] and not INCLUDE_GPT:
            continue
        else:
            rows_new.append(r)

    table = tabulate(rows_new, headers=headers, tablefmt="github")
    print(table)


def create_table_winogender(task_triple, rows):
    headers = ["Model"]
    task, metric_name, direction = task_triple
    # print(task, metric_name, direction)
    # header = task + f" ({metric_name.replace('accuracy', 'acc')}) " + ("↑" if direction == "max" else "↓")
    header = task
    headers.append(header + " (Parity)")
    headers.append(header + " (Alpha)")

    rows_new = []
    for r in rows:
        if "gpt" in r[0] and not INCLUDE_GPT:
            continue
        else:
            rows_new.append(r)

    table = tabulate(rows_new, headers=headers, tablefmt="github")
    print(table)


def create_table_hyperparameters(task_triples, model_and_task_to_hps):
    headers = ["Task", "LR", "BS", "hps std"]
    # tasks = [t for t, _, _ in task_triples]
    model_to_stds = defaultdict(list)
    model_to_mean_distances = defaultdict(list)
    model_tuples = [(m, m_dict) for m, m_dict in model_and_task_to_hps.items()]
    model_tuples.sort(key=lambda x: x[0])
    for model, model_dict in model_tuples:
        if "gpt" in model and not INCLUDE_GPT:
            continue
        # print(model)
        rows = []
        for task, metric_name, direction in task_triples:
            row = [task]
            # print(model, model_dict)
            # WD = 0 if gpt else 0.1
            # num_epochs = 10 (early stopping, patience 5)
            # warmup_ratio = 0.06
            # fp16 = True
            task_dict = model_dict[task]
            lr = task_dict["learning_rate"]
            bs = task_dict["batch_size"]
            std = task_dict["hps_analysis"]["std"]
            metrics = task_dict["hps_analysis"]["metrics"]
            model_to_stds[model].append(std)
            model_to_mean_distances[model].append(avg_distance_from_max(metrics))
            row.append(lr)
            row.append(bs)
            row.append(std)

            rows.append(row)

        table = tabulate(rows, headers=headers, tablefmt="github")
        print("*" * 50)
        print("**" + model + "**\n")
        print(table)

    # Avg Std table
    headers = ["Model", "avg std"]
    rows = []
    for model, stds in model_to_stds.items():
        rows.append([model, np.mean(stds)])

    rows.sort(key=lambda x: x[-1])
    table = tabulate(rows, headers=headers, tablefmt="github")
    print("*" * 50)
    print(table)

    # Avg mean distance to max table
    headers = ["Model", "avg mean distance"]
    rows = []
    for model, mean_distances in model_to_mean_distances.items():
        print(model, mean_distances)
        rows.append([model, np.mean(mean_distances)])

    rows.sort(key=lambda x: x[-1])
    table = tabulate(rows, headers=headers, tablefmt="github")
    print("*" * 50)
    print(table)


def main():
    model_dicts, task_triples, model_and_task_to_hps = create_and_load_model_dicts()
    task_triple_winogender = [tt for tt in task_triples if tt[0] == "SweWinogender"][0]
    task_triples = [tt for tt in task_triples if tt[0] != "SweWinogender"]
    sorted_model_tuples = sorted([(k, v) for k, v in model_dicts.items()], key=lambda x: x[0])
    sorted_model_dicts = [model_dict for _, model_dict in sorted_model_tuples]

    rows = [create_row(model_dict, task_triples) for model_dict in sorted_model_dicts]
    rows.sort(key=lambda x: x[-1], reverse=True)

    rows_winogender = [create_row(model_dict, [task_triple_winogender]) for model_dict in sorted_model_dicts]
    rows_winogender = [r for r in rows_winogender if r[-1] is not None]
    rows_winogender.sort(key=lambda x: x[-1], reverse=True)

    create_table(task_triples, rows)
    print("*" * 50)
    create_table_winogender(task_triple_winogender, rows_winogender)
    print("*" * 50)
    create_table_hyperparameters(task_triples, model_and_task_to_hps)
    # print(model_and_task_to_hps)


if __name__ == '__main__':
    main()
