from typing import Dict
from sklearn.dummy import DummyRegressor, DummyClassifier

from Experiment import Experiment

is_task_regression = {
    "SweParaphrase": True,
    "SweFAQ": False,
    "DaLAJ": False,
    "ABSAbank-Imm": True,
}


class ExperimentDummy(Experiment):

    def __init__(self, task_name: str):
        model_name = "Dummy"
        data_fraction = 1.0
        super().__init__(task_name, model_name, data_fraction)

    @staticmethod
    def _train_regression(train_labels):
        regressor = DummyRegressor(strategy="mean")
        regressor.fit(range(len(train_labels)), train_labels)
        return regressor

    @staticmethod
    def _train_classification(train_labels):
        classifier = DummyClassifier(strategy="most_frequent")
        classifier.fit(range(len(train_labels)), train_labels)
        return classifier

    def _evaluate(self, model, dev_labels, test_labels):
        predictions_eval = model.predict(range(len(dev_labels)))
        predictions_test = model.predict(range(len(test_labels)))
        eval_score = self._compute_metrics((predictions_eval, dev_labels))
        test_score = self._compute_metrics((predictions_test, test_labels))

        metric_dict = {
            "eval": eval_score[self.metric],
            "test": test_score[self.metric]
        }

        return metric_dict

    def run_impl(self) -> Dict[str, float]:
        train_ds, dev_ds, test_ds = self._load_data()
        train_labels = train_ds["labels"]
        dev_labels = dev_ds["labels"]
        test_labels = test_ds["labels"]

        is_regression = is_task_regression[self.task_name]
        if is_regression:
            model = self._train_regression(train_labels)
        else:
            model = self._train_classification(train_labels)

        return self._evaluate(model, dev_labels, test_labels)
