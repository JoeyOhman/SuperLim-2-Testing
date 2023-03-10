from bert.ExperimentBert import ExperimentBert


class ExperimentBertReviews(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool,
                 evaluate_only: bool):
        task_name = "Reviews"
        # max_input_length = 128
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run, evaluate_only)

    def preprocess_data(self, dataset_split):
        dataset_split = dataset_split.map(
            lambda sample: self.batch_tokenize(sample['text']), batched=True, num_proc=4)
            # lambda sample: tokenizer(sample['text'], truncation=True, max_length=max_len),
            # batched=True, num_proc=4)

        # dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['label'])}, batched=False, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)
        dataset_split = dataset_split.remove_columns(['text'])

        for sample in dataset_split:
            print(sample)
            break
        #     exit()

        return dataset_split
