import csv
import os
import random
from collections import defaultdict

from datasets import load_dataset, Dataset

from paths import TSV_PATH

# DATASET_PATH = os.path.join(JSONL_PATH, "SweParaphrase")
# DATASET_PATH_TEMPLATE = os.path.join(JSONL_PATH, "{task}")
DATASET_PATH_TEMPLATE = os.path.join(TSV_PATH, "{task}")


def reformat_dataset_SweFAQ(dataset):
    new_dataset_dict = {
        "question": [],
        "answer": [],
        "labels": []
    }

    categories = defaultdict(list)
    for sample in dataset:
        cat_id = sample["category_id"]
        categories[cat_id].append(sample)

    # Category probably not needed
    for category, samples in categories.items():
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

    return Dataset.from_dict(new_dataset_dict)


def reformat_dataset_DaLAJ(dataset):
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


def load_dataset_by_task(task_name: str, data_fraction: float = 1.0):
    # data_files = glob.glob(DATASET_PATH + "*.tsv")
    dataset_path = DATASET_PATH_TEMPLATE.format(task=task_name)
    data_files = {
        "train": os.path.join(dataset_path, "train.tsv"),
        "dev": os.path.join(dataset_path, "dev.tsv")
    }

    # dataset = load_dataset('json', data_files=data_files)
    # dataset = load_dataset('csv', data_files=data_files, delimiter="\t")
    dataset = load_dataset('csv', data_files=data_files, delimiter="\t", quoting=csv.QUOTE_NONE)
    train_ds, dev_ds = dataset['train'], dataset['dev']

    # Reformat raw data to training tasks
    task_to_reformat_fun = {
        "SweFAQ": reformat_dataset_SweFAQ,
        "DaLAJ": reformat_dataset_DaLAJ
    }
    format_fun = task_to_reformat_fun.get(task_name, None)
    if format_fun is not None:
        train_ds = format_fun(train_ds)
        dev_ds = format_fun(dev_ds)

    # For debugging: split dev set into dev and fake-test
    num_test = int(len(dev_ds) / 3)
    break_point_idx = len(dev_ds) - num_test
    test_ds = dev_ds.select(range(break_point_idx, len(dev_ds)))
    dev_ds = dev_ds.select(range(0, break_point_idx))

    assert 0.0 < data_fraction <= 1.0, "data_fraction must be in range(0, 1]"
    if data_fraction < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * data_fraction)))
        dev_ds = dev_ds.select(range(int(len(dev_ds) * data_fraction)))
        test_ds = test_ds.select(range(int(len(test_ds) * data_fraction)))

    return train_ds, dev_ds, test_ds


if __name__ == '__main__':
    load_dataset_by_task("SweParaphrase", 1.0)
