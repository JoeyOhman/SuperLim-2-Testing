import random

import evaluate
import krippendorff
import numpy as np
from sklearn.metrics import mean_squared_error


# def get_compute_metrics_fun(metric_name: str):
#     fun_name = "_compute_metrics_" + metric_name
#     getattr()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1", )
spearmanr_metric = evaluate.load("spearmanr")


def _ensure_flattened(list_or_array):
    if isinstance(list_or_array[0], list) or isinstance(list_or_array[0], np.ndarray):
        list_or_array = [item for sublist in list_or_array for item in sublist]

    return list_or_array


def _compute_metrics_rmse(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    # predictions = np.argmax(predictions, axis=1)
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def _compute_metrics_normalized_spearmanr(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    # predictions = np.argmax(predictions, axis=1)
    spearman_r = spearmanr_metric.compute(references=labels, predictions=predictions)
    spearman_r["spearmanr"] = (spearman_r["spearmanr"] + 1) / 2
    return spearman_r


def _compute_metrics_rmse_and_normalized_spearmanr(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    # predictions = np.argmax(predictions, axis=1)
    rmse = mean_squared_error(labels, predictions, squared=False)
    spearman_r = spearmanr_metric.compute(references=labels, predictions=predictions)
    spearman_r["spearmanr"] = (spearman_r["spearmanr"] + 1) / 2
    combined_dict = {"rmse": rmse}
    combined_dict.update(spearman_r)
    return combined_dict


def _compute_metrics_accuracy(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    if np.ndim(predictions) > 1:
        predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(references=labels, predictions=predictions)
    # return {"accuracy": acc}
    return acc


def _compute_metrics_f1(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    if np.ndim(predictions) > 1:
        predictions = np.argmax(predictions, axis=1)
    # Macro gives equal weight to each class, i.e. not taking into account class imbalance
    f1 = f1_metric.compute(references=labels, predictions=predictions, average='macro')
    # return {"f1": f1}
    return f1


def _compute_metrics_krippendorff(preds_labels_tuple, level_of_measurement):
    predictions, labels = preds_labels_tuple
    if np.ndim(predictions) > 1 and level_of_measurement != "interval":
        predictions = np.argmax(predictions, axis=1)
    predictions = _ensure_flattened(predictions)
    labels = _ensure_flattened(labels)
    # level_of_measurement is:
    # classification: nominal
    # regression: interval
    # arr = np.array([predictions, labels])
    # alpha = krippendorff.alpha(reliability_data=arr, level_of_measurement=level_of_measurement)
    alpha = krippendorff.alpha(reliability_data=[predictions, labels], level_of_measurement=level_of_measurement)
    metric_name = f"krippendorff_{level_of_measurement}"
    return {metric_name: alpha}


def _compute_metrics_krippendorff_classification(preds_labels_tuple):
    alpha = _compute_metrics_krippendorff(preds_labels_tuple, "nominal")
    acc = _compute_metrics_accuracy(preds_labels_tuple)
    alpha.update(acc)
    return alpha


def _compute_metrics_krippendorff_regression(preds_labels_tuple):
    alpha = _compute_metrics_krippendorff(preds_labels_tuple, "interval")
    rmse = _compute_metrics_rmse(preds_labels_tuple)
    alpha.update(rmse)
    return alpha


metric_to_compute_fun = {
    # "rmse": _compute_metrics_rmse,
    "rmse": _compute_metrics_rmse_and_normalized_spearmanr,
    "accuracy": _compute_metrics_accuracy,
    "f1": _compute_metrics_f1,
    "spearmanr": _compute_metrics_normalized_spearmanr,
    "rmse_spearmanr": _compute_metrics_rmse_and_normalized_spearmanr,
    "krippendorff_nominal": _compute_metrics_krippendorff_classification,
    "krippendorff_interval": _compute_metrics_krippendorff_regression
}

if __name__ == '__main__':
    preds = [2.3, 3., 5., 4.]
    # preds = [[2.3], [3.], [5.], [4.]]
    # labels = np.array([3, 2, 6, 4.4])
    labels = [3, 2, 6, 4.4]
    # {'rmse': 0.8139410298049854, 'spearmanr': 0.8999999999999999}
    # res = _compute_metrics_rmse_and_normalized_spearmanr((preds, labels))
    # res = _compute_metrics_rmse_and_normalized_spearmanr((preds, labels))
    # {'krippendorff': 0.8268135561572215}
    # preds = [1, 2, 3, 4]
    # labels = [1, 2, 3, 5]
    # preds_wrapped = [[p] for p in preds]
    # preds = [random.random() for _ in range(500)]
    # labels = [random.random() for _ in range(500)]
    print(len(preds))
    res = _compute_metrics_krippendorff_regression([preds, labels])
    # {'krippendorff_interval': -0.15348706936793954, 'rmse': 0.8139410298049854}
    # {'krippendorff_interval': 0.8268135561572215, 'rmse': 0.8139410298049854}
    print("Res:", res)
    # res_wrapped = _compute_metrics_krippendorff_regression((preds_wrapped, labels))
    # print("Res_wrapped:", res_wrapped)
    # res = _compute_metrics_krippendorff_classification((preds, labels))
    # print(res)
