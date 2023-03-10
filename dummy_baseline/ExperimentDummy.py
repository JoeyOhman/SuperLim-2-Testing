import random
from typing import Dict, Tuple
from sklearn.dummy import DummyRegressor, DummyClassifier

from Experiment import Experiment
from dataset_loaders.dataset_loader import load_dataset_by_task, reformat_eval_set_swefaq


class ExperimentDummy(Experiment):

    def __init__(self, task_name: str, is_random=False):
        model_name = "Random" if is_random else "MaxFreq/Avg"
        data_fraction = 1.0
        self.is_random = is_random
        super().__init__(task_name, model_name, data_fraction, evaluate_only=False)

    def _train_regression(self, train_labels):
        regressor = None
        if self.is_random:
            self.min_val = min(train_labels)
            self.max_val = max(train_labels)
            #  regressor = DummyRegressor(strategy="constant", )
            # regressor.fit(range(len(train_labels)), train_labels)
        else:
            regressor = DummyRegressor(strategy="mean")
            regressor.fit(range(len(train_labels)), train_labels)
        return regressor

    def _train_classification(self, train_labels):
        classifier = DummyClassifier(strategy="uniform" if self.is_random else "most_frequent")
        classifier.fit(range(len(train_labels)), train_labels)
        return classifier

    def _evaluate(self, model, dev_labels, test_labels):

        if self.is_regression and self.is_random:
            predictions_eval = [random.uniform(self.min_val, self.max_val) for _ in range(len(dev_labels))]
            predictions_test = [random.uniform(self.min_val, self.max_val) for _ in range(len(test_labels))]
        else:
            predictions_eval = model.predict(range(len(dev_labels)))
            predictions_test = model.predict(range(len(test_labels)))

        eval_score = self._compute_metrics((predictions_eval, dev_labels))
        test_score = self._compute_metrics((predictions_test, test_labels))

        metric_dict = {
            "eval": eval_score[self.metric],
            "test": test_score[self.metric]
        }

        return metric_dict, predictions_eval, predictions_test

    def run_impl(self) -> Tuple[Dict[str, float], list, list]:
        train_ds, dev_ds, test_ds = self._load_data()
        train_labels = train_ds["labels"]
        dev_labels = dev_ds["labels"]
        test_labels = test_ds["labels"]

        if self.task_name == "SweFAQ":
            _, val_ds_raw, test_ds_raw = load_dataset_by_task(self.task_name, self.data_fraction, reformat=False)
            val_ds = reformat_eval_set_swefaq(val_ds_raw)
            test_ds = reformat_eval_set_swefaq(test_ds_raw)
            dev_labels = val_ds["labels"]
            test_labels = test_ds["labels"]

        if self.is_regression:
            model = self._train_regression(train_labels)
        else:
            model = self._train_classification(train_labels)

        return self._evaluate(model, dev_labels, test_labels)
