import os
import glob
from datasets import load_dataset

from paths import JSONL_PATH

DATASET_PATH = os.path.join(JSONL_PATH, "SweParaphrase")


def load_swe_paraphrase(data_fraction=1.0):
    # data_files = glob.glob(DATASET_PATH + "*.tsv")
    data_files = {
        "train": os.path.join(DATASET_PATH, "train.jsonl"),
        "dev": os.path.join(DATASET_PATH, "dev.jsonl")
    }

    dataset = load_dataset('json', data_files=data_files)
    train_ds, dev_ds = dataset['train'], dataset['dev']

    # For debugging: split dev set into dev and fake-test
    num_test = int(len(dev_ds) / 2)
    test_ds = dev_ds.select(range(num_test, len(dev_ds)))
    dev_ds = dev_ds.select(range(0, num_test))

    assert 0.0 < data_fraction <= 1.0, "data_fraction must be in range(0, 1]"
    if data_fraction < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * data_fraction)))
        dev_ds = dev_ds.select(range(int(len(dev_ds) * data_fraction)))
        test_ds = test_ds.select(range(int(len(test_ds) * data_fraction)))

    return train_ds, dev_ds, test_ds


if __name__ == '__main__':
    load_swe_paraphrase(1.0)
