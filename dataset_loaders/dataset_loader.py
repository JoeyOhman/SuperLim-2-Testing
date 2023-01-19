import csv
import os
import random
from collections import defaultdict

from datasets import load_dataset, Dataset
from pathlib import Path

from paths import TSV_PATH, DATASET_PATH

# DATASET_PATH_TEMPLATE = os.path.join(JSONL_PATH, "{task}")
DATASET_PATH_TEMPLATE = os.path.join(DATASET_PATH, "{task}")


def reformat_dataset_SweParaphrase(dataset, split_name):
    dataset = dataset.rename_column("Score", "labels")
    return dataset


def reformat_dataset_ABSAbankImm(dataset, split_name):
    dataset = dataset.rename_column("label", "labels")
    return dataset


def reformat_dataset_SweFAQ(dataset, split_name):
    new_dataset_dict = {
        "question": [],
        "answer": [],
        "labels": []
    }

    # Group samples by categories
    categories = defaultdict(list)
    for sample in dataset:
        cat_id = sample["category_id"]
        categories[cat_id].append(sample)

    # Category probably not needed
    for category, samples in categories.items():
        # if split_name == "train":
        for idx, sample in enumerate(samples):
            # Positive sample
            q, a = sample["question"], sample["correct_answer"]
            new_dataset_dict["question"].append(q)
            new_dataset_dict["answer"].append(a)
            new_dataset_dict["labels"].append(1)

            # Negative samples (might need to under sample here)
            other_indices = [i for i in range(len(samples)) if i != idx]
            choice = random.choice(other_indices)

            incorrect_answer = samples[choice]["correct_answer"]
            new_dataset_dict["question"].append(q)
            new_dataset_dict["answer"].append(incorrect_answer)
            new_dataset_dict["labels"].append(0)
            # for other_idx, other_sample in enumerate(samples):
            #     if idx == other_idx:
            #         continue
            #     incorrect_answer = other_sample["correct_answer"]
            #     new_dataset_dict["question"].append(q)
            #     new_dataset_dict["answer"].append(incorrect_answer)
            #     new_dataset_dict["labels"].append(0)
        # else:
        #     # dev/test
        #     questions = [sample["question"] for sample in samples]
        #     answers = [sample["correct_answer"] for sample in samples]
        #
        #     # For each question, add all answers in the category
        #     for idx, question in enumerate(questions):
        #         new_dataset_dict["question"].append(question)
        #         new_dataset_dict["answer"].append(answers)
        #         new_dataset_dict["labels"].append(idx)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_DaLAJ(dataset, split_name):
    new_dataset_dict = {
        "text": [],
        "labels": []
    }
    for sample in dataset:
        new_dataset_dict["text"].append(sample["original sentence"])
        new_dataset_dict["labels"].append(0)
        new_dataset_dict["text"].append(sample["corrected sentence"])
        new_dataset_dict["labels"].append(1)

    return Dataset.from_dict(new_dataset_dict)


def reformat_eval_set_swefaq(ds_split):
    new_dataset_dict = {
        "question": [],
        "answer": [],
        "labels": []
    }

    # Group samples by categories
    categories = defaultdict(list)
    for sample in ds_split:
        cat_id = sample["category_id"]
        categories[cat_id].append(sample)

    for category, samples in categories.items():
        questions = [sample["question"] for sample in samples]
        answers = [sample["correct_answer"] for sample in samples]

        # For each question, add all answers in the category
        for idx, question in enumerate(questions):
            new_dataset_dict["question"].append(question)
            new_dataset_dict["answer"].append(answers)
            new_dataset_dict["labels"].append(idx)

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
        label = 1 if sample["same_sense"] else 0
        new_dataset_dict["labels"].append(label)

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_Reviews(dataset, split_name):
    dataset = dataset.rename_column("label", "labels")
    return dataset


TASK_TO_REFORMAT_FUN = {
    "SweParaphrase": reformat_dataset_SweParaphrase,
    "SweFAQ": reformat_dataset_SweFAQ,
    "DaLAJ": reformat_dataset_DaLAJ,
    "ABSAbank-Imm": reformat_dataset_ABSAbankImm,
    "SweWiC": reformat_dataset_SweWiC,
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
    if data_fraction < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * data_fraction)))
        dev_ds = dev_ds.select(range(int(len(dev_ds) * data_fraction)))
        test_ds = test_ds.select(range(int(len(test_ds) * data_fraction)))

    return train_ds, dev_ds, test_ds


if __name__ == '__main__':
    dataset_main = load_dataset_by_task("SweWiC", 1.0)
    print(dataset_main)
    # ds = load_dataset("super_glue", "boolq")
    # print(ds)
