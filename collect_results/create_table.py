import numpy as np
from tabulate import tabulate

from collect_results.collect_model_metrics import create_and_load_model_dicts

TEST_SCORES = True


def create_row(model_dict, task_triples):
    model_name = model_dict["model"]["model_name"]
    row = [model_name]
    for task, _, _ in task_triples:
        # if task == "SweWinogender":
        #     continue
        task_dict = model_dict["tasks"][task]
        row.append(task_dict["test" if TEST_SCORES else "dev"])

    scores = [col if task_triples[idx][-1] == "max" else -col for idx, col in enumerate(row[1:])]
    scores = [-100 if score is None else score for score in scores]
    # agg_column = sum(scores)
    agg_column = np.mean(scores)
    row.append(agg_column)
    return row


def create_table(task_triples, rows):
    headers = ["Model"]
    for task, metric_name, direction in task_triples:
        # print(task, metric_name, direction)
        header = task + f" ({metric_name.replace('accuracy', 'acc')}) " + ("↑" if direction == "max" else "↓")
        headers.append(header)
    headers.append("Avg ↑")

    table = tabulate(rows, headers=headers, tablefmt="github")
    print(table)


def main():
    model_dicts, task_triples = create_and_load_model_dicts()
    task_triples = [tt for tt in task_triples if tt[0] != "SweWinogender"]
    sorted_model_tuples = sorted([(k, v) for k, v in model_dicts.items()], key=lambda x: x[0])
    sorted_model_dicts = [model_dict for _, model_dict in sorted_model_tuples]

    rows = [create_row(model_dict, task_triples) for model_dict in sorted_model_dicts]
    rows.sort(key=lambda x: x[-1], reverse=True)

    create_table(task_triples, rows)


if __name__ == '__main__':
    main()
