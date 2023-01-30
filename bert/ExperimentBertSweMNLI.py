import json
from collections import defaultdict

import transformers
import numpy as np
from pathlib import Path

from bert.ExperimentBert import ExperimentBert
from dataset_loaders.dataset_loader import load_winogender
from paths import get_experiment_metrics_path


class ExperimentBertSweMNLI(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweMNLI"
        # max_input_length = 128
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

    def preprocess_data(self, dataset_split):
        # Map works sample by sample or in batches if batched=True
        dataset_split = dataset_split.map(
            lambda sample: self.batch_tokenize(sample['sentence1'], sample['sentence2']), batched=True, num_proc=1)

        # dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['labels'])}, batched=False, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)

        dataset_split = dataset_split.remove_columns(['sentence1', 'sentence2'])

        # for sample in dataset_split:
        #     print(sample)
        #     break

        return dataset_split

    def _predict_winogender_sample(self, model, sentence1, sentence2, tuple_id):
        encoded = self.batch_tokenize([sentence1], [sentence2], ret_tensors="pt")
        encoded = encoded.to(self.device)
        logits = model(**encoded).logits[0].cpu().detach().numpy()
        prediction = np.argmax(logits)
        return prediction

    def _predict_winogender(self, model, test_ds):
        predictions, labels, tuple_ids = [], [], []
        for sample in test_ds:
            sentence1, sentence2, tuple_id = sample["sentence1"], sample["sentence2"], sample["tuple_id"]
            prediction = self._predict_winogender_sample(model, sentence1, sentence2, tuple_id)
            predictions.append(prediction)
            labels.append(sample["labels"])
            tuple_ids.append(tuple_id)

        return predictions, labels, tuple_ids

    def _calculate_parity(self, predictions, tuple_ids):
        # Group by tuple_id
        tuple_id_to_prediction = defaultdict(list)
        for prediction, tuple_id in zip(predictions, tuple_ids):
            tuple_id_to_prediction[tuple_id].append(prediction)

        # Define helper for checking if all elements in list are equal
        def all_equal(iterator):
            return len(set(iterator)) <= 1

        # Calculate number of groups & number of groups with all equal predictions
        num_groups = 0
        num_same = 0
        for tuple_id, predictions_in_group in tuple_id_to_prediction.items():
            num_groups += 1
            if all_equal(predictions_in_group):
                num_same += 1

        # Calculate parity
        parity = num_same / num_groups
        return parity

    # Override entire _evaluate in order to squeeze in SweWinogender evaluation
    def _evaluate(self, trainer, val_ds, test_ds):
        # Get original result first for SweMNLI, to avoid disrupting the trainer
        dict_to_return = super()._evaluate(trainer, val_ds, test_ds)

        # Load SweWinogender
        winogender_test_ds = load_winogender()
        # winogender_test_ds = self.preprocess_data(winogender_test_ds)

        # Get predictions
        transformers.logging.set_verbosity_error()
        model = trainer.model
        model.to(self.device)
        model.eval()
        predictions, labels, tuple_ids = self._predict_winogender(model, winogender_test_ds)

        # num_correct = 0
        # num_tot = len(predictions)
        # for prediction, label, tuple_id in zip(predictions, labels, tuple_ids):
        #     if prediction == label:
        #         num_correct += 1

        metrics = self._compute_metrics((predictions, labels))
        parity = self._calculate_parity(predictions, tuple_ids)

        # Pack results
        metric_dict = {
            "task": "SweWinogender",
            "model": self.model_name,
            "metric": "parity",
            "direction": self.direction,
            "parity": parity,
            self.metric: metrics[self.metric],
            "accuracy": metrics["accuracy"],
            "hyperparameters": self.hp_dict
        }

        # Dump to file
        experiment_metric_path = get_experiment_metrics_path("SweWinogender", self.model_name)
        Path(experiment_metric_path).mkdir(parents=True, exist_ok=True)
        metric_str = json.dumps(metric_dict, ensure_ascii=False, indent="\t")
        print(metric_str)
        with open(experiment_metric_path + "/metrics.json", 'w') as f:
            f.write(metric_str)

        return dict_to_return


if __name__ == '__main__':
    bert_swepara = ExperimentBertSweMNLI("albert-base-v2", 1.0, False, True)
    bert_swepara.run()
