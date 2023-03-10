import csv
import os
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path

from paths import TSV_PATH, DATASET_PATH

# DATASET_PATH_TEMPLATE = os.path.join(JSONL_PATH, "{task}")
DATASET_PATH_TEMPLATE = os.path.join(DATASET_PATH, "{task}")


def convert_label_to_original(task, label):
    if task == "DaLAJ":
        if label == 1:
            return "correct"
        elif label == 0:
            return "incorrect"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    elif task == "SweWiC":
        if label == 0:
            return "different_sense"
        elif label == 1:
            return "same_sense"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    elif task == "SweMNLI":
        if label == 0:
            return "entailment"
        elif label == 1:
            return "contradiction"
        elif label == 2:
            return "neutral"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    elif task == "SweWinograd":
        if label == 1:
            return "coreferring"
        elif label == 0:
            return "not_coreferring"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    elif task == "SweWinogender":
        if label == 0:
            return "entailment"
        elif label == 1:
            return "contradiction"
        elif label == 2:
            return "neutral"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    # SweFAQ should be in a good format already
    elif task == "ArgumentationSentences":
        if label == 0:
            return "con"
        elif label == 1:
            return "pro"
        elif label == 2:
            return "non"
        else:
            print(f"LABEL ERROR for task={task}, label={label}")
            assert False

    else:
        return label


def reformat_dataset_ArgumentationSentences(dataset, split_name):
    new_dataset_dict = {
        "sentence1": [],
        "sentence2": [],
        "labels": []
    }

    for sample in dataset:
        new_dataset_dict["sentence1"].append(sample["sentence"])
        new_dataset_dict["sentence2"].append(sample["topic"])

        if sample["label"] == "con":
            label = 0
        elif sample["label"] == "pro":
            label = 1
        elif sample["label"] == "non":
            label = 2
        else:
            print(f"label incorrect in data file {split_name}:", sample["label"])
            assert False

        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_SweParaphrase(dataset, split_name):
    # dataset = dataset.rename_column("Score", "labels")
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.rename_column("sentence_1", "Sentence 1")
    dataset = dataset.rename_column("sentence_2", "Sentence 2")
    return dataset


def reformat_dataset_ABSAbankImm(dataset, split_name):
    dataset = dataset.rename_column("label", "labels")
    return dataset


# def reformat_dataset_SweFAQ(dataset, split_name):
#     new_dataset_dict = {
#         "question": [],
#         "answer": [],
#         "labels": []
#     }
#
#     # Group samples by categories
#     categories = defaultdict(list)
#     for sample in dataset:
#         cat_id = sample["category_id"]
#         categories[cat_id].append(sample)
#
#     # Category probably not needed
#     for category, samples in categories.items():
#
#         for idx, sample in enumerate(samples):
#             # Positive sample
#             q, a = sample["question"], sample["correct_answer"]
#             new_dataset_dict["question"].append(q)
#             new_dataset_dict["answer"].append(a)
#             new_dataset_dict["labels"].append(1)
#
#             # Negative samples (might need to under sample here)
#             other_indices = [i for i in range(len(samples)) if i != idx]
#             choice = random.choice(other_indices)
#
#             incorrect_answer = samples[choice]["correct_answer"]
#             new_dataset_dict["question"].append(q)
#             new_dataset_dict["answer"].append(incorrect_answer)
#             new_dataset_dict["labels"].append(0)
#
#     return Dataset.from_dict(new_dataset_dict)
#
#
# def reformat_eval_set_swefaq(ds_split):
#     new_dataset_dict = {
#         "question": [],
#         "answer": [],
#         "labels": []
#     }
#
#     # Group samples by categories
#     categories = defaultdict(list)
#     for sample in ds_split:
#         cat_id = sample["category_id"]
#         categories[cat_id].append(sample)
#
#     for category, samples in categories.items():
#         questions = [sample["question"] for sample in samples]
#         answers = [sample["correct_answer"] for sample in samples]
#
#         # For each question, add all answers in the category
#         for idx, question in enumerate(questions):
#             new_dataset_dict["question"].append(question)
#             new_dataset_dict["answer"].append(answers)
#             new_dataset_dict["labels"].append(idx)
#
#     return Dataset.from_dict(new_dataset_dict)

def reformat_dataset_SweFAQ(dataset, split_name):
    new_dataset_dict = {
        "question": [],
        "answer": [],
        "labels": []
    }

    for sample in dataset:
        question = sample["question"]
        correct_index = sample["label"]
        correct_answer = sample["candidate_answers"][correct_index]
        other_indices = [i for i in range(len(sample["candidate_answers"])) if i != correct_index]
        incorrect_index = random.choice(other_indices)
        incorrect_answer = sample["candidate_answers"][incorrect_index]

        # Positive sample
        new_dataset_dict["question"].append(question)
        new_dataset_dict["answer"].append(correct_answer)
        new_dataset_dict["labels"].append(1)

        # Negative sample
        new_dataset_dict["question"].append(question)
        new_dataset_dict["answer"].append(incorrect_answer)
        new_dataset_dict["labels"].append(0)

    return Dataset.from_dict(new_dataset_dict)


def reformat_eval_set_swefaq(ds_split):
    new_dataset_dict = {
        "question": [],
        "answer": [],
        "labels": []
    }

    for sample in ds_split:
        question = sample["question"]
        label = sample["label"]
        answers = sample["candidate_answers"]

        assert len(answers) >= label - 1

        new_dataset_dict["question"].append(question)
        new_dataset_dict["answer"].append(answers)
        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_DaLAJ(dataset, split_name):
    new_dataset_dict = {
        "text": [],
        "labels": []
    }

    # OLD
    # for sample in dataset:
    #     new_dataset_dict["text"].append(sample["original sentence"])
    #     new_dataset_dict["labels"].append(0)
    #     new_dataset_dict["text"].append(sample["corrected sentence"])
    #     new_dataset_dict["labels"].append(1)

    for sample in dataset:
        new_dataset_dict["text"].append(sample["sentence"])

        if sample["label"] == "correct":
            label = 1
        elif sample["label"] == "incorrect":
            label = 0
        else:
            print("Incorrect label in data file:", sample["label"])
            assert False

        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_SweWiC(dataset, split_name):
    new_dataset_dict = {
        "sentence1": [],
        "sentence2": [],
        "word1": [],
        "word2": [],
        "labels": []
    }

    for sample in dataset:
        new_dataset_dict["sentence1"].append(sample["first"]["context"])
        new_dataset_dict["sentence2"].append(sample["second"]["context"])
        new_dataset_dict["word1"].append(sample["first"]["word"]["text"])
        new_dataset_dict["word2"].append(sample["second"]["word"]["text"])
        label = 1 if sample["label"] == "same_sense" else 0
        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_SweWinograd(dataset, split_name):
    new_dataset_dict = {
        "text": [],
        "pronoun": [],
        "candidate": [],
        "labels": []
    }

    for sample in dataset:
        new_dataset_dict["text"].append(sample["text"])
        new_dataset_dict["pronoun"].append(sample["pronoun"]["text"])
        new_dataset_dict["candidate"].append(sample["candidate_antecedent"]["text"])

        if sample["label"] == "coreferring":
            label = 1
        elif sample["label"] == "not_coreferring":
            label = 0
        else:
            print("label incorrect in data file:", sample["label"])
            assert False

        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_SweMNLI(dataset, split_name):
    new_dataset_dict = {
        "sentence1": [],
        "sentence2": [],
        "labels": []
    }

    for sample in dataset:
        new_dataset_dict["sentence1"].append(sample["premise"])
        new_dataset_dict["sentence2"].append(sample["hypothesis"])

        if sample["label"] == "entailment":
            label = 0
        elif sample["label"] == "contradiction":
            label = 1
        elif sample["label"] == "neutral":
            label = 2
        else:
            print(f"label incorrect in data file {split_name}:", sample["label"])
            assert False

        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_Reviews(dataset, split_name):
    dataset = dataset.rename_column("label", "labels")
    return dataset


TASK_TO_REFORMAT_FUN = {
    "ArgumentationSentences": reformat_dataset_ArgumentationSentences,
    "SweParaphrase": reformat_dataset_SweParaphrase,
    "SweFAQ": reformat_dataset_SweFAQ,
    "DaLAJ": reformat_dataset_DaLAJ,
    "ABSAbank-Imm": reformat_dataset_ABSAbankImm,
    "SweWiC": reformat_dataset_SweWiC,
    "SweWinograd": reformat_dataset_SweWinograd,
    "SweMNLI": reformat_dataset_SweMNLI,
    "Reviews": reformat_dataset_Reviews,
}


def load_dataset_by_task(task_name: str, data_fraction: float = 1.0, from_hf: bool = False, reformat: bool = True):

    if task_name == "Reviews":
        dataset = load_dataset("swedish_reviews")
        train_ds, dev_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']
    elif from_hf:
        # From HuggingFace Datasets
        hf_task_name = task_name.lower()
        if hf_task_name == "absabank-imm":
            hf_task_name = "absabank"
        elif hf_task_name == "sweparaphrase":
            hf_task_name = "swepar"
        # dataset = load_dataset(f"sbx/superlim-2/{task_name}")
        dataset = load_dataset("sbx/superlim-2", hf_task_name)
    else:
        # From local TSV
        # data_files = glob.glob(DATASET_PATH + "*.tsv")
        dataset_path = DATASET_PATH_TEMPLATE.format(task=task_name)
        # data_files = glob.glob(DATASET_PATH + "train.*")
        extension = [pt for pt in Path(dataset_path).rglob("train.*")][0].name.split(".")[-1]
        is_json = "json" in extension

        data_files = {
            "train": os.path.join(dataset_path, f"train.{extension}"),
            "dev": os.path.join(dataset_path, f"dev.{extension}"),
            "test": os.path.join(dataset_path, f"test.{extension}")
        }

        # dataset = load_dataset('json', data_files=data_files)
        # dataset = load_dataset('csv', data_files=data_files, delimiter="\t")
        if is_json:
            dataset = load_dataset('json', data_files=data_files)
        else:
            dataset = load_dataset('csv', data_files=data_files, delimiter="\t", quoting=csv.QUOTE_NONE)

    if task_name != "Reviews":
        train_ds, dev_ds, test_ds = dataset['train'], dataset['dev'], dataset['test']
        # train_ds, dev_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']

    # Reformat raw data to training tasks
    if reformat:
        format_fun = TASK_TO_REFORMAT_FUN.get(task_name, None)
        if format_fun is not None:
            train_ds = format_fun(train_ds, "train")
            dev_ds = format_fun(dev_ds, "dev")
            test_ds = format_fun(test_ds, "test")

    # For debugging: split dev set into dev and fake-test
    # num_test = int(len(dev_ds) / 3)
    # break_point_idx = len(dev_ds) - num_test
    # test_ds = dev_ds.select(range(break_point_idx, len(dev_ds)))
    # dev_ds = dev_ds.select(range(0, break_point_idx))

    assert 0.0 < data_fraction <= 1.0, "data_fraction must be in range(0, 1]"
    if data_fraction < 1.0:  # For debugging purposes!
        train_ds = train_ds.select(range(int(len(train_ds) * data_fraction)))
        dev_ds = dev_ds.select(range(int(len(dev_ds) * data_fraction)))
        test_ds = test_ds.select(range(int(len(test_ds) * data_fraction)))

    if task_name == "SweMNLI":  # For reducing immense training set
        df = 1.0
        if df < 1.0:
            train_ds = train_ds.shuffle(seed=42)
            train_ds = train_ds.select(range(int(len(train_ds) * df)))

    return train_ds, dev_ds, test_ds


def load_winogender(reformat=True):
    dataset_path = DATASET_PATH_TEMPLATE.format(task="SweWinogender")
    extension = [pt for pt in Path(dataset_path).rglob("test.*")][0].name.split(".")[-1]
    is_json = "json" in extension
    data_files = {"test": os.path.join(dataset_path, f"test.{extension}")}
    if is_json:
        dataset = load_dataset('json', data_files=data_files)
    else:
        dataset = load_dataset('csv', data_files=data_files, delimiter="\t", quoting=csv.QUOTE_NONE)

    new_dataset_dict = {
        "sentence1": [],
        "sentence2": [],
        "labels": [],
        "tuple_id": []
    }

    test_ds = dataset["test"]
    if not reformat:
        return test_ds

    for sample in test_ds:
        new_dataset_dict["sentence1"].append(sample["premise"])
        new_dataset_dict["sentence2"].append(sample["hypothesis"])

        if sample["label"] == "entailment":
            label = 0
        elif sample["label"] == "contradiction":
            label = 1
        elif sample["label"] == "neutral":
            label = 2
        else:
            print("label incorrect in data file:", sample["label"])
            assert False

        new_dataset_dict["labels"].append(label)
        new_dataset_dict["tuple_id"].append(sample["meta"]["tuple_id"])

    return Dataset.from_dict(new_dataset_dict)


if __name__ == '__main__':
    # dataset_main = load_dataset_by_task("SweWiC", 1.0)
    dataset_main = load_winogender()
    print(dataset_main)
    # ds = load_dataset("super_glue", "boolq")
    # print(ds)
