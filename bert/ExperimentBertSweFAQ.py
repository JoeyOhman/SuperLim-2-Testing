from collections import defaultdict
from typing import Dict

import torch
import transformers
import numpy as np
from datasets import Dataset

from bert.ExperimentBert import ExperimentBert
from compute_metrics import _compute_metrics_accuracy
from dataset_loaders.dataset_loader import load_dataset_by_task, reformat_eval_set_swefaq


class ExperimentBertSweFAQ(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool,
                 evaluate_only: bool):
        task_name = "SweFAQ"
        # max_input_length = 256
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run, evaluate_only)

    def preprocess_data(self, dataset_split):
        dataset_split = dataset_split.map(
            lambda sample: self.batch_tokenize(sample["question"], sample["answer"]), batched=True, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)
        # dataset_split = dataset_split.remove_columns(['question', 'answer'])

        # for sample in dataset_split:
        #     ids = sample["input_ids"].detach().cpu().numpy()
        #     mask = sample["attention_mask"].detach().cpu().numpy()
        #     print(ids)
        #     print(mask)
        #     print(self.tokenizer.decode(ids, skip_special_tokens=False))
        #     exit()

        return dataset_split

    def _predict_sample_cross(self, model, question, answers):
        probs = []
        binary_preds = []
        for answer in answers:
            # encoded = tokenizer(question, answer, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
            encoded = self.batch_tokenize([question], [answer], ret_tensors="pt")
            encoded = encoded.to(self.device)
            logits = model(**encoded).logits[0].cpu().detach().numpy()
            probs.append(logits[1])
            binary_preds.append(np.argmax(logits))

        prediction = np.argmax(probs)
        return prediction, binary_preds

    def _predict_dataset_cross(self, tokenizer, model, dataset):
        predictions, labels = [], []
        binary_correct = []
        binary_num_yes = 0
        binary_num_tot = 0
        for sample in dataset:
            question, answers, label = sample["question"], sample["answer"], sample["labels"]
            # print("Question:")
            # print(question)
            # print("Answers:")
            # print("\n\n".join([str(i) + ": " + a for i, a in enumerate(answers)]))
            # print("Label:")
            # print(label)
            # print("Correct answer:")
            # print(answers[label])
            # exit()
            prediction, binary_preds = self._predict_sample_cross(model, question, answers)

            for idx, bin_pred in enumerate(binary_preds):
                if label == idx:  # this should be 1
                    correct = bin_pred == 1
                else:  # this should be 0
                    correct = bin_pred == 0

                if bin_pred == 1:
                    binary_num_yes += 1
                binary_num_tot += 1

                binary_correct.append(1 if correct else 0)

            predictions.append(prediction)
            labels.append(label)

        binary_accuracy = sum(binary_correct) / len(binary_correct)
        print("****************Binary accuracy:", binary_accuracy)
        print("################Binary fraction yes:", binary_num_yes / binary_num_tot)

        metric_dict = self._compute_metrics((predictions, labels))
        return metric_dict, predictions

    # Override entire _evaluate in order to do cross encoder
    def _evaluate(self, trainer, val_ds, test_ds):
        # tokenizer = self._load_tokenizer()
        transformers.logging.set_verbosity_error()
        model = trainer.model
        model.to(self.device)
        model.eval()

        _, val_ds_raw, test_ds_raw = load_dataset_by_task(self.task_name, self.data_fraction, reformat=False)
        val_ds = reformat_eval_set_swefaq(val_ds_raw)
        test_ds = reformat_eval_set_swefaq(test_ds_raw)

        val_metric, predictions_val = self._predict_dataset_cross(self.tokenizer, model, val_ds)
        test_metric, predictions_test = self._predict_dataset_cross(self.tokenizer, model, test_ds)

        # Pack results
        # assert self.metric == "accuracy"
        metric_dict = {
            "eval": val_metric[self.metric],
            "test": test_metric[self.metric],
            "eval_metrics_obj": val_metric,
            "test_metrics_obj": test_metric
        }

        return metric_dict, predictions_val, predictions_test
