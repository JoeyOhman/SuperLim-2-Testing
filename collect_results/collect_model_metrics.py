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

DATA_SIZES = {
    "KB/bert-base-swedish-cased": "20GB",
    "KBLab/megatron-bert-base-swedish-cased-600k": "70GB",
    "KBLab/bert-base-swedish-cased-new": "70GB",
    "xlm-roberta-base": "2500GB",
    "AI-Nordics/bert-large-swedish-cased": "85GB",
    "KBLab/megatron-bert-large-swedish-cased-165k": "70GB",
    "xlm-roberta-large": "2500GB",
    "NbAiLab/nb-bert-base": "109GB"
}


def load_model_cards():
    model_cards_path = "/home/joey/code/ai/SuperLim-2-Testing/data/ModelCards"
    model_cards_dict = {}
    for path in Path(model_cards_path).rglob('*.txt'):
        with open(path, 'r') as f:
            model_card = f.read()

        if "AI-Nordics-bert-large-swedish-cased" in path.name:
            model_name = "AI-Nordics/bert-large-swedish-cased"
        else:
            model_name = path.name.split("/")[-1].split(".txt")[0]
            if "xlm-roberta" not in model_name:
                model_name = model_name.replace("-", "/", 1)

        model_cards_dict[model_name] = model_card

    print(model_cards_dict)

    return model_cards_dict


MODEL_CARDS = load_model_cards()


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


def get_model_card(model_name):
    model_card = MODEL_CARDS.get(model_name, None)
    if model_card is not None:
        return model_card

    if model_name == "Decision Tree":
        model_card = "The model was trained on with the vectorizer performing best on the dev-set of the following:\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html"
    elif model_name == "SVM":
        model_card = "The model was trained on with the vectorizer performing best on the dev-set of the following:\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"
    elif model_name == "Random Forest":
        model_card = "The model was trained on with the vectorizer performing best on the dev-set of the following:\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"
    elif model_name == "MaxFreq/Avg":
        model_card = "This model uses Max Frequency for classification, and the mean strategy for regression\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html\n" \
                     "https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html"
    elif model_name == "Random":
        model_card = "This model predicts a random output in the possible range observed in the training set."
    else:
        if "gpt" in model_name:
            return None
        print(model_name)
        assert False

    return model_card


def get_init_model_dict(model, tasks):
    model_type = "Non-transformer"
    if "bert" in model:
        model_type = "Encoder"
    if "gpt" in model:
        model_type = "Decoder"


    def init_task_dict(task):
        if task == "SweWinogender":
            return {
                "dev": None,
                "test": {"alpha": None, "parity": None}
            }
        else:
            return {
                "dev": None,
                "test": None
            }

    return {
        "repo_link": "https://github.com/JoeyOhman/SuperLim-2-Testing",
        "model": {
            "model_name": model,
            "model_type": model_type,
            "number_params": MODEL_SIZES.get(model, None),
            "data_size": DATA_SIZES.get(model, None)
        },
        "tasks": {
            TASK_NAME_MAP.get(task, task): init_task_dict(task) for task in TASK_NAMES
        },
        # "model_card": MODEL_CARDS.get(model, None)
        "model_card": get_model_card(model)
    }


def add_task_to_model_dict(model_dicts, model, task, metric_name, dev, test):
    if task == "SweWinogender":
        model_dicts[model]["tasks"][task]["test"] = {
            "alpha": dev,
            "parity": test
        }
    else:
        model_dicts[model]["tasks"][task] = {
            # "metric": metric_name,
            "dev": dev,
            "test": test
        }


def create_and_save_model_dicts(metric_dicts, tasks):
    model_dicts = defaultdict(dict)
    model_and_task_to_hps = defaultdict(dict)
    for metric_dict in metric_dicts:
        # Extract properties
        task = metric_dict["task"]
        task = TASK_NAME_MAP.get(task, task)
        model = metric_dict["model"]
        metric_name = metric_dict["metric"]
        if task == "SweWinogender":
            dev = metric_dict["krippendorff_nominal"]
            test = metric_dict["parity"]
        else:
            dev = metric_dict["eval"]
            test = metric_dict["test"]

        # Initialize and fill model dict
        if model not in model_dicts:
            model_dicts[model] = get_init_model_dict(model, tasks)
        add_task_to_model_dict(model_dicts, model, task, metric_name, dev, test)

        if "hyperparameters" in metric_dict:
            model_and_task_to_hps[model][task] = metric_dict["hyperparameters"]

    # Write one json file per model
    deliverables_path = METRICS_PATH + "/model_deliverables"
    Path(deliverables_path).mkdir(parents=True, exist_ok=True)
    for model, model_dict in model_dicts.items():
        with open(deliverables_path + "/" + model.replace("/", "-") + ".json", 'w') as f:
            json.dump(model_dict, f, ensure_ascii=False, indent="\t")

    return model_dicts, model_and_task_to_hps


def create_and_load_model_dicts():
    metric_dicts = load_all_metric_dicts()
    task_triples = get_all_tasks(metric_dicts)
    model_dicts, model_and_task_to_hps = create_and_save_model_dicts(metric_dicts, task_triples)
    return model_dicts, task_triples, model_and_task_to_hps


if __name__ == '__main__':
    create_and_load_model_dicts()
