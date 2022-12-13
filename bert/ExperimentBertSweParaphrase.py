from bert.ExperimentBert import ExperimentBert


class ExperimentBertSweParaphrase(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweParaphrase"
        # max_input_length = 128
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

    @staticmethod
    def preprocess_data(dataset_split, tokenizer, max_len, dataset_split_name):
        # Map works sample by sample or in batches if batched=True
        dataset_split = dataset_split.map(
            lambda sample: tokenizer(sample['Sentence 1'], sample['Sentence 2'], truncation=True,
                                     max_length=max_len),
            batched=True, num_proc=4)

        # Could use batched=True here and do float conversion in list comprehension
        # dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['Score'])}, batched=False, num_proc=4)
        dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['labels'])}, batched=False, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)

        dataset_split = dataset_split.remove_columns(['Genre', 'File', 'Sentence 1', 'Sentence 2'])
        # dataset_split = dataset_split.remove_columns(['Genre', 'File', 'Sentence 1', 'Sentence 2', 'Score'])
        return dataset_split


if __name__ == '__main__':
    bert_swepara = ExperimentBertSweParaphrase("albert-base-v2", 1.0, False, True)
    bert_swepara.run()
