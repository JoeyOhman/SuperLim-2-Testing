import csv
import os
import glob
from datasets import load_dataset

from paths import JSONL_PATH, TSV_PATH

# DATASET_PATH = os.path.join(JSONL_PATH, "SweParaphrase")
# DATASET_PATH_TEMPLATE = os.path.join(JSONL_PATH, "{task}")
DATASET_PATH_TEMPLATE = os.path.join(TSV_PATH, "{task}")


def load_dataset_by_task(task_name: str, data_fraction: float = 1.0):
    # data_files = glob.glob(DATASET_PATH + "*.tsv")
    dataset_path = DATASET_PATH_TEMPLATE.format(task=task_name)
    data_files = {
        "train": os.path.join(dataset_path, "train.tsv"),
        "dev": os.path.join(dataset_path, "dev.tsv")
    }

    # dataset = load_dataset('json', data_files=data_files)
    dataset = load_dataset('csv', data_files=data_files, delimiter="\t", quoting=csv.QUOTE_NONE)
    train_ds, dev_ds = dataset['train'], dataset['dev']

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
