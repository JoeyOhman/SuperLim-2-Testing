import json
from collections import defaultdict
from pathlib import Path

from paths import METRICS_PATH
from store_model_sizes import load_model_sizes

TASK_NAMES = [
    "DaLAJ",
    "Swedish FAQ",
    # "SweFraCas",
    "SweMNLI",
    "SweWinograd",
    "SweWinogender",
    "SweDiagnostics",
    "SweParaphrase",
    "SweWiC",
    "SweSAT",
    "Swedish Analogy",
    "SuperSim",
    "ABSA",
]

TASK_NAME_MAP = {
    "ABSAbank-Imm": "ABSA",
    "SweFAQ": "Swedish FAQ"
}

MODEL_SIZES = load_model_sizes()


def load_all_metric_dicts():
    metric_dicts = []
    for path in Path(METRICS_PATH).rglob('metrics.json'):
        with open(path, 'r') as f:
            json_dict = json.load(f)

        metric_dicts.append(json_dict)

    return metric_dicts


def get_all_tasks(metric_dicts):
    tasks = {}
    for metric_dict in metric_dicts:
        task = metric_dict["task"]
        task = TASK_NAME_MAP.get(task, task)
        direction = metric_dict["direction"]
        metric_name = metric_dict["metric"]
        tasks[task] = (metric_name, direction)

    triples = []
    for task, (metric_name, direction) in tasks.items():
        triples.append((task, metric_name, direction))
    triples.sort(key=lambda x: x[0])
    return triples


def get_init_model_dict(model, tasks):
    model_type = "Non-transformer"
    if "bert" in model:
        model_type = "Encoder"
    if "gpt" in model:
        model_type = "Decoder"

    return {
        "repo_link": "https://github.com/JoeyOhman/SuperLim-2-Testing",
        "model": {
            "model_name": model,
            "model_type": model_type,
            "number_params": MODEL_SIZES.get(model, None),
            "data_size": None
        },
        "tasks": {
            TASK_NAME_MAP.get(task, task): {
                # "metric": metric_name,
                # "dev": 100 if direction == "min" else -100,
                # "test": 100 if direction == "min" else -100
                "dev": None,
                "test": None
            # } for task, metric_name, direction in tasks
            } for task in TASK_NAMES
        },
        "model_card": ""
    }


def add_task_to_model_dict(model_dicts, model, task, metric_name, dev, test):
    model_dicts[model]["tasks"][task] = {
        # "metric": metric_name,
        "dev": dev,
        "test": test
    }


def create_and_save_model_dicts(metric_dicts, tasks):
    model_dicts = defaultdict(dict)
    for metric_dict in metric_dicts:
        # Extract properties
        task = metric_dict["task"]
        task = TASK_NAME_MAP.get(task, task)
        model = metric_dict["model"]
        metric_name = metric_dict["metric"]
        dev = metric_dict["eval"]
        test = metric_dict["test"]

        # Initialize and fill model dict
        if model not in model_dicts:
            model_dicts[model] = get_init_model_dict(model, tasks)
        add_task_to_model_dict(model_dicts, model, task, metric_name, dev, test)

    # Write one json file per model
    deliverables_path = METRICS_PATH + "/model_deliverables"
    Path(deliverables_path).mkdir(parents=True, exist_ok=True)
    for model, model_dict in model_dicts.items():
        with open(deliverables_path + "/" + model.replace("/", "-") + ".json", 'w') as f:
            json.dump(model_dict, f, ensure_ascii=False, indent="\t")

    return model_dicts


def create_and_load_model_dicts():
    metric_dicts = load_all_metric_dicts()
    task_triples = get_all_tasks(metric_dicts)
    model_dicts = create_and_save_model_dicts(metric_dicts, task_triples)
    return model_dicts, task_triples


if __name__ == '__main__':
    create_and_load_model_dicts()
