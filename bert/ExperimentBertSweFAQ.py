from collections import defaultdict
from typing import Dict

import torch
import numpy as np
from datasets import Dataset

from bert.ExperimentBert import ExperimentBert
from compute_metrics import _compute_metrics_accuracy
from dataset_loaders.dataset_loader import load_dataset_by_task


class ExperimentBertSweFAQ(ExperimentBert):

    def __init__(self, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool, quick_run: bool):
        task_name = "SweFAQ"
        # max_input_length = 256
        super().__init__(task_name, model_name, accumulation_steps, data_fraction, hps, quick_run)

    @staticmethod
    def preprocess_data(dataset_split, tokenizer, max_len, dataset_split_name):
        dataset_split = dataset_split.map(
            lambda sample: tokenizer(sample['question'], sample['answer'], truncation=True, max_length=max_len),
            batched=True, num_proc=4)

        features = list(dataset_split.features.keys())
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in features:
            columns.append('token_type_ids')
        dataset_split.set_format(type='torch', columns=columns)
        dataset_split = dataset_split.remove_columns(['question', 'answer'])

        return dataset_split

    @staticmethod
    def _reformat_eval_sets(ds_split):
        new_dataset_dict = {
            "question": [],
            "answer": [],
            "labels": []
        }

        # Group samples by categories
        categories = defaultdict(list)
        for sample in ds_split:
            cat_id = sample["category_id"]
            categories[cat_id].append(sample)

        for category, samples in categories.items():
            questions = [sample["question"] for sample in samples]
            answers = [sample["correct_answer"] for sample in samples]

            # For each question, add all answers in the category
            for idx, question in enumerate(questions):
                new_dataset_dict["question"].append(question)
                new_dataset_dict["answer"].append(answers)
                new_dataset_dict["labels"].append(idx)

        return Dataset.from_dict(new_dataset_dict)

    def _predict_sample_cross(self, tokenizer, model, question, answers):
        probs = []
        for answer in answers:
            encoded = tokenizer(question, answer, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
            encoded = encoded.to(self.device)
            logits = model(**encoded).logits[0].cpu().detach().numpy()
            probs.append(logits[1])

        prediction = np.argmax(probs)
        return prediction

    def _predict_dataset_cross(self, tokenizer, model, dataset):
        predictions, labels = [], []
        for sample in dataset:
            question, answers, label = sample["question"], sample["answer"], sample["labels"]
            prediction = self._predict_sample_cross(tokenizer, model, question, answers)
            predictions.append(prediction)
            labels.append(label)

        acc = _compute_metrics_accuracy((predictions, labels))
        return acc

    # Override entire _evaluate in order to do cross encoder
    def _evaluate(self, trainer, val_ds, test_ds):
        tokenizer = self._load_tokenizer()
        model = trainer.model
        model.to(self.device)
        model.eval()

        _, val_ds_raw, test_ds_raw = load_dataset_by_task(self.task_name, 1.0, reformat=False)
        val_ds = self._reformat_eval_sets(val_ds_raw)
        test_ds = self._reformat_eval_sets(test_ds_raw)

        val_metric = self._predict_dataset_cross(tokenizer, model, val_ds)
        test_metric = self._predict_dataset_cross(tokenizer, model, test_ds)

        # Pack results
        assert self.metric == "accuracy"
        metric_dict = {
            "eval": val_metric[self.metric],
            "test": test_metric[self.metric],
            "eval_metrics_obj": val_metric,
            "test_metrics_obj": test_metric
        }

        return metric_dict
