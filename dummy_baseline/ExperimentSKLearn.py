import random
from typing import Dict

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from Experiment import Experiment
from dataset_loaders.dataset_loader import load_dataset_by_task, reformat_eval_set_swefaq

SUPPORTED_MODELS = ["Decision Tree", "SVM", "Random Forest"]


# https://github.com/RobinQrtz/superlim_baselines/blob/main/simple_baselines.py
class ExperimentSKLearn(Experiment):

    def __init__(self, task_name: str, model_name: str):
        assert model_name in SUPPORTED_MODELS
        data_fraction = 1.0
        self.sep = " [SEP] "
        super().__init__(task_name, model_name, data_fraction)

    def find_best_vectorizer(self, model_class, texts_train, labels_train, texts_dev, labels_dev):
        vectorizers = []
        ngram_ranges = [(1, 2), (1, 3)]
        for ngram_range in ngram_ranges:
            vectorizers.append(CountVectorizer(ngram_range=ngram_range, lowercase=False, binary=False))
            vectorizers.append(TfidfVectorizer(ngram_range=ngram_range, lowercase=False, binary=False))

        pipelines = []
        dev_scores = []
        for vectorizer in vectorizers:
            print("Testing vectorizer for:", self.model_name)
            if self.model_name == "Random Forest":
                model_instance = model_class(max_depth=2, n_estimators=500, random_state=0, min_samples_split=10,
                                             min_samples_leaf=2, n_jobs=-1, oob_score=True)
            else:
                model_instance = model_class()
            model = make_pipeline(vectorizer, model_instance)
            model.fit(texts_train, labels_train)

            predictions_eval = model.predict(texts_dev)
            dev_score = self._compute_metrics((predictions_eval, labels_dev))

            pipelines.append(model)
            dev_scores.append(dev_score[self.metric])

        best_idx = np.argmax(dev_scores) if self.direction == "max" else np.argmin(dev_scores)
        return pipelines[best_idx]

    def _train_regression(self, texts_train, labels_train, texts_dev, labels_dev):
        if self.model_name == "Decision Tree":
            model_class = DecisionTreeRegressor
        elif self.model_name == "SVM":
            model_class = LinearSVR
        elif self.model_name == "Random Forest":
            model_class = RandomForestRegressor
        else:
            raise NotImplementedError(f"Select a supported model_name, not supported: {self.model_name}")

        # model = make_pipeline(self.vectorizer, model_class())
        # model.fit(texts_train, labels_train)
        model = self.find_best_vectorizer(model_class, texts_train, labels_train, texts_dev, labels_dev)
        return model

    def _train_classification(self, texts_train, labels_train, texts_dev, labels_dev):
        if self.model_name == "Decision Tree":
            model_class = DecisionTreeClassifier
        elif self.model_name == "SVM":
            model_class = LinearSVC
        elif self.model_name == "Random Forest":
            model_class = RandomForestClassifier
        else:
            raise NotImplementedError(f"Select a supported model_name, not supported: {self.model_name}")

        # model = make_pipeline(self.vectorizer, model_class())
        # model.fit(texts_train, labels_train)
        model = self.find_best_vectorizer(model_class, texts_train, labels_train, texts_dev, labels_dev)
        return model

    def _predict_sample_cross(self, model, question, answers):
        preds = []
        for answer in answers:
            pred = model.predict([question + self.sep + answer])[0]
            preds.append(pred)

        one_indices = []
        for i, pred in enumerate(preds):
            if pred == 1:
                one_indices.append(i)

        if len(one_indices) > 0:
            prediction = random.choice(one_indices)
        else:
            prediction = random.choice(preds)

        return prediction

    def _predict_dataset_cross(self, model, dataset):
        predictions, labels = [], []
        for sample in dataset:
            question, answers, label = sample["question"], sample["answer"], sample["labels"]
            prediction = self._predict_sample_cross(model, question, answers)
            predictions.append(prediction)
            labels.append(label)

        return predictions, labels

    def _evaluate(self, model, texts_dev, labels_dev, texts_test, labels_test):

        if self.task_name == "SweFAQ":
            _, val_ds_raw, test_ds_raw = load_dataset_by_task(self.task_name, self.data_fraction, reformat=False)
            val_ds = reformat_eval_set_swefaq(val_ds_raw)
            test_ds = reformat_eval_set_swefaq(test_ds_raw)

            predictions_eval, labels_dev = self._predict_dataset_cross(model, val_ds)
            predictions_test, labels_test = self._predict_dataset_cross(model, test_ds)
        else:
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
            model = self._train_regression(texts_train, labels_train, texts_dev, labels_dev)
        else:
            model = self._train_classification(texts_train, labels_train, texts_dev, labels_dev)

        return self._evaluate(model, texts_dev, labels_dev, texts_test, labels_test)

    # TODO: other tasks
    def get_raw_text_and_labels(self, dataset_split):
        if self.task_name == "DaLAJ":
            return dataset_split["text"], dataset_split["labels"]
        elif self.task_name == "ABSAbank-Imm":
            return dataset_split["text"], dataset_split["labels"]
        elif self.task_name == "SweFAQ":
            return [s1 + self.sep + s2 for s1, s2 in zip(dataset_split["question"], dataset_split["answer"])], dataset_split["labels"]
        elif self.task_name == "SweParaphrase":
            return [s1 + self.sep + s2 for s1, s2 in zip(dataset_split["Sentence 1"], dataset_split["Sentence 2"])], dataset_split["labels"]
        elif self.task_name == "SweWiC":
            return [s1 + self.sep + s2 + self.sep + w1 + self.sep + w2 for s1, s2, w1, w2 in zip(dataset_split["sentence1"], dataset_split["sentence2"], dataset_split["word1"], dataset_split["word2"])], dataset_split["labels"]
        elif self.task_name == "SweWinograd":
            return [t + self.sep + p + self.sep + c for t, p, c in zip(dataset_split["text"], dataset_split["pronoun"], dataset_split["candidate"])], dataset_split["labels"]
        else:
            print(f"Task={self.task_name} reformatting function for raw text not implemented")
            exit()


if __name__ == '__main__':
    print("SweFAQ:")
    # ExperimentSKLearn("SweFAQ", "SVM").run()
    ExperimentSKLearn("SweFAQ", "Decision Tree").run()
    # print("ABSAbank-Imm:")
    # ExperimentDecisionTree("ABSAbank-Imm").run()
    # print("SweParaphrase:")
    # ExperimentDecisionTree("SweParaphrase").run()
