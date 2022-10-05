import evaluate
import numpy as np
from sklearn.metrics import mean_squared_error


# def get_compute_metrics_fun(metric_name: str):
#     fun_name = "_compute_metrics_" + metric_name
#     getattr()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1", )


def _compute_metrics_rmse(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    # predictions = np.argmax(predictions, axis=1)
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def _compute_metrics_accuracy(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(references=labels, predictions=predictions)
    # return {"accuracy": acc}
    return acc


def _compute_metrics_f1(preds_labels_tuple):
    predictions, labels = preds_labels_tuple
    predictions = np.argmax(predictions, axis=1)
    # Macro gives equal weight to each class, i.e. not taking into account class imbalance
    f1 = f1_metric.compute(references=labels, predictions=predictions, average='macro')
    # return {"f1": f1}
    return f1


metric_to_compute_fun = {
    "rmse": _compute_metrics_rmse,
    "accuracy": _compute_metrics_accuracy,
    "f1": _compute_metrics_f1
}
