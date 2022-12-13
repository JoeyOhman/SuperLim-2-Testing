from bert.ExperimentBert import ExperimentBert


class ExperimentBertSweFAQ(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweFAQ"
        # max_input_length = 256
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

    @staticmethod
    def preprocess_data(dataset_split, tokenizer, max_len, dataset_split_name):
        if dataset_split_name == "train":
            dataset_split = dataset_split.map(
                lambda sample: tokenizer(sample['question'], sample['answer'], truncation=True, max_length=max_len),
                batched=True, num_proc=4)

            features = list(dataset_split.features.keys())
            columns = ['input_ids', 'attention_mask', 'labels']
            if 'token_type_ids' in features:
                columns.append('token_type_ids')
            dataset_split.set_format(type='torch', columns=columns)
            dataset_split = dataset_split.remove_columns(['question', 'answer'])
        else:
            # sample: (question, answers_list, label_idx)

            # dataset_split = dataset_split.map(lambda sample: {"input_ids_question": tokenizer()})
            # dataset_split = dataset_split.map(
            #     lambda sample: tokenizer(sample['question'], truncation=True, max_length=max_len),
            #     batched=True, num_proc=4)
            pass

        return dataset_split

    def _predict(self, model, val_ds, test_ds):

        model.eval()

        for sample in val_ds:
            question, answers, label = sample["question"], sample["answer"], sample["labels"]
            print(question)
            print(answers)
            print(label)
            exit()
        # for ds in val_ds, test_ds:

        return None
