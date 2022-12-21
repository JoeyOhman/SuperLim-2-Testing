from typing import Dict

from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

from Experiment import Experiment


# TODO: vectorize, look here: https://github.com/RobinQrtz/superlim_baselines/blob/main/simple_baselines.py
#  Create inputs for each dataset as it depends on format?
class ExperimentDecisionTree(Experiment):

    def __init__(self, task_name: str):
        model_name = "Decision Tree"
        data_fraction = 1.0
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=False, binary=False)
        super().__init__(task_name, model_name, data_fraction)

    def _train_regression(self, texts_train, labels_train, texts_dev, labels_dev, texts_test, labels_test):
        model = make_pipeline(self.vectorizer, DecisionTreeRegressor())
        model.fit(texts_train, labels_train)
        return model

    def _train_classification(self, texts_train, labels_train, texts_dev, labels_dev, texts_test, labels_test):
        model = make_pipeline(self.vectorizer, DecisionTreeClassifier())
        model.fit(texts_train, labels_train)
        return model

    def _evaluate(self, model, texts_dev, labels_dev, texts_test, labels_test):
        predictions_eval = model.predict(texts_dev)
        predictions_test = model.predict(texts_test)
        eval_score = self._compute_metrics((predictions_eval, labels_dev))
        test_score = self._compute_metrics((predictions_test, labels_test))

        metric_dict = {
            "eval": eval_score[self.metric],
            "test": test_score[self.metric]
        }

        return metric_dict

    def run_impl(self) -> Dict[str, float]:
        train_ds, dev_ds, test_ds = self._load_data()
        # train_labels = train_ds["labels"]
        # dev_labels = dev_ds["labels"]
        # test_labels = test_ds["labels"]
        texts_train, labels_train = self.get_raw_text_and_labels(train_ds)
        texts_dev, labels_dev = self.get_raw_text_and_labels(dev_ds)
        texts_test, labels_test = self.get_raw_text_and_labels(test_ds)

        if self.is_regression:
            model = self._train_regression(texts_train, labels_train, texts_dev, labels_dev, texts_test, labels_test)
        else:
            model = self._train_classification(texts_train, labels_train, texts_dev, labels_dev, texts_test, labels_test)

        return self._evaluate(model, texts_dev, labels_dev, texts_test, labels_test)

    # TODO: other tasks
    def get_raw_text_and_labels(self, dataset_split):
        if self.task_name == "DaLAJ":
            return dataset_split["text"], dataset_split["labels"]
        elif self.task_name == "ABSAbank-Imm":
            return dataset_split["text"], dataset_split["labels"]
        elif self.task_name == "SweFAQ":
            pass
        elif self.task_name == "SweParaphrase":
            pass
        else:
            print(f"Task={self.task_name} reformatting function for raw text not implemented")
            exit()


if __name__ == '__main__':
    print("DaLAJ:")
    ExperimentDecisionTree("DaLAJ").run()
    print("ABSAbank-Imm:")
    ExperimentDecisionTree("ABSAbank-Imm").run()
