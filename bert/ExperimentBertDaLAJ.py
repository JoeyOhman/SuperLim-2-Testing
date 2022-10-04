from collections import Counter

from bert.ExperimentBert import ExperimentBert


class ExperimentBertDaLAJ(ExperimentBert):

    def __init__(self, model_name: str, data_fraction: float, quick_run: bool):
        task_name = "DaLAJ"
        max_input_length = 128
        super().__init__(task_name, model_name, data_fraction, max_input_length, quick_run)

    @staticmethod
    def _pre_process_data(dataset_split, tokenizer, max_len):
        pass

    def create_dataset(self, tokenizer, max_seq_len: int):
        train_ds, dev_ds, test_ds = self._load_data()
        # TODO: reformat entire dataset, where to do this? this will be the same for all kinds of models, and may
        #  be relevant for future datasets/tasks

        print(train_ds)
        counter = 0
        for sample in train_ds:
            # print("*" * 50)
            # print(sample["original sentence"])
            # print(sample["corrected sentence"])
            assert sample["original sentence"] != sample["corrected sentence"]
            # print(sample["error label"])
            counter += 1
            # if counter >= 20:
            #     break

        labels = []
        for sample in train_ds:
            labels.append(sample["error label"])
        print("train:", Counter(labels))

        labels = []
        for sample in dev_ds:
            labels.append(sample["error label"])
        print("dev:", Counter(labels))
        exit()
