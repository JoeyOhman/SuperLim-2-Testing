import random
from collections import defaultdict

from datasets import Dataset
import numpy as np

from bert.ExperimentBert import ExperimentBert


class ExperimentBertSweFAQ(ExperimentBert):

    def __init__(self, model_name: str, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweFAQ"
        max_input_length = 256
        super().__init__(task_name, model_name, data_fraction, max_input_length, hps, quick_run)

    @staticmethod
    def _pre_process_data(dataset_split, tokenizer, max_len):
        dataset_split = dataset_split.map(
            lambda sample: tokenizer(sample['question'], sample['answer'], truncation=True, max_length=max_len),
            batched=True, num_proc=4)

        dataset_split.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset_split

    @staticmethod
    def _reformat_data(dataset_split):
        new_dataset_dict = {
            "question": [],
            "answer": [],
            "labels": []
        }

        categories = defaultdict(list)
        for sample in dataset_split:
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
