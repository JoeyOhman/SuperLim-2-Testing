from bert.ExperimentBert import ExperimentBert


class ExperimentBertSweWiC(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweWiC"
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

    def preprocess_data(self, dataset_split):
        # Map works sample by sample or in batches if batched=True
        dataset_split = dataset_split.map(
            # lambda sample: self.batch_tokenize(sample['sentence1'], sample['sentence2']),
            lambda sample: self.batch_tokenize(sample['sentence1'], sample['sentence2'], sample['word1'], sample['word2']),
            batched=True, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)

        dataset_split = dataset_split.remove_columns(['sentence1', 'sentence2', 'word1', 'word2'])

        # for sample in dataset_split:
        #     print("AFTER TOKENIZATION")
        #     print("input_ids:")
        #     print(sample["input_ids"])
        #     print("decoded:")
        #     print(self.tokenizer.decode(sample['input_ids']))
        #     print("label:")
        #     print(sample["labels"])
        #     break
            # exit()

        return dataset_split

