from collections import Counter

from datasets import Dataset
import numpy as np

from bert.ExperimentBert import ExperimentBert


class ExperimentBertDaLAJ(ExperimentBert):

    def __init__(self, model_name: str, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "DaLAJ"
        max_input_length = 128
        super().__init__(task_name, model_name, data_fraction, max_input_length, hps, quick_run)

    @staticmethod
    def _pre_process_data(dataset_split, tokenizer, max_len):
        dataset_split = dataset_split.map(
            lambda sample: tokenizer(sample['text'], truncation=True, max_length=max_len),
            batched=True, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)

        return dataset_split

    @staticmethod
    def _reformat_data(dataset_split):

        new_dataset_dict = {
            "text": [],
            "labels": []
        }
        for sample in dataset_split:
            new_dataset_dict["text"].append(sample["original sentence"])
            new_dataset_dict["labels"].append(0)
            new_dataset_dict["text"].append(sample["corrected sentence"])
            new_dataset_dict["labels"].append(1)

        return Dataset.from_dict(new_dataset_dict)

    def create_dataset(self, tokenizer, max_seq_len: int):
        train_ds, dev_ds, test_ds = self._load_data()

        train_ds = self._reformat_data(train_ds)
        dev_ds = self._reformat_data(dev_ds)
        test_ds = self._reformat_data(test_ds)

        print("Preprocessing train_ds")
        train_ds = self._pre_process_data(train_ds, tokenizer, max_seq_len)
        print("Preprocessing dev_ds")
        dev_ds = self._pre_process_data(dev_ds, tokenizer, max_seq_len)
        print("Preprocessing test_ds")
        test_ds = self._pre_process_data(test_ds, tokenizer, max_seq_len)

        train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
        num_max_len = sum([1 if length == self.max_input_length else 0 for length in train_ds_lens])
        print("Num samples that are max length:", num_max_len)
        print("Train ds, Max len:", max(train_ds_lens))
        print("Train ds, Mean len:", np.mean(train_ds_lens))
        print(f"#samples:\ntrain={train_ds.num_rows}, dev={dev_ds.num_rows}, test={test_ds.num_rows}")

        return train_ds, dev_ds, test_ds
