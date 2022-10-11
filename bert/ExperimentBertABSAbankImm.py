import numpy as np
from bert.ExperimentBert import ExperimentBert


class ExperimentBertABSAbankImm(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "ABSAbank-Imm"
        # max_input_length = 256
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

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

        dataset_split = dataset_split.remove_columns(
            ['id', 'text', 'a0', 'a1', 'a3', 'a4', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11'])

        return dataset_split

    def create_dataset(self, tokenizer, max_seq_len: int):
        train_ds, dev_ds, test_ds = self._load_data()

        print("Preprocessing train_ds")
        train_ds = self._pre_process_data(train_ds, tokenizer, max_seq_len)
        print("Preprocessing dev_ds")
        dev_ds = self._pre_process_data(dev_ds, tokenizer, max_seq_len)
        print("Preprocessing test_ds")
        test_ds = self._pre_process_data(test_ds, tokenizer, max_seq_len)

        train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
        print("Train ds, Max len:", max(train_ds_lens))
        print("Train ds, Mean len:", np.mean(train_ds_lens))
        print(f"#samples:\ntrain={train_ds.num_rows}, dev={dev_ds.num_rows}, test={test_ds.num_rows}")

        return train_ds, dev_ds, test_ds
