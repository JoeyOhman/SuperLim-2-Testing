from typing import Dict
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from Experiment import Experiment


# TODO: vectorize, look here: https://github.com/RobinQrtz/superlim_baselines/blob/main/simple_baselines.py
    #  Create inputs for each dataset as it depends on format?
class ExperimentDecisionTree(Experiment):

    def __init__(self, task_name: str):
        model_name = "Decision Tree"
        data_fraction = 1.0
        super().__init__(task_name, model_name, data_fraction)

    @staticmethod
    def _train_regression(train_inputs, train_labels):
        regressor = DecisionTreeRegressor()
        regressor.fit(train_inputs, train_labels)
        return regressor

    @staticmethod
    def _train_classification(train_inputs, train_labels):
        classifier = DecisionTreeClassifier()
        classifier.fit(train_inputs, train_labels)
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

        if self.is_regression:
            model = self._train_regression(train_labels)
        else:
            model = self._train_classification(train_labels)

        return self._evaluate(model, dev_labels, test_labels)