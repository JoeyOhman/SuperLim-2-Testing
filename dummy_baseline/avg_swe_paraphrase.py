import numpy as np

from sklearn.metrics import mean_squared_error

from dataset_loaders.data_loader_swe_paraphrase import load_swe_paraphrase


def compute_rmse(predictions, labels):
    return mean_squared_error(labels, predictions, squared=False)


if __name__ == '__main__':
    train_ds, dev_ds = load_swe_paraphrase()
    train_scores = [float(sample['Score']) for sample in train_ds]
    dev_scores = [float(sample['Score']) for sample in dev_ds]
    mean_score = np.mean(train_scores)
    # print(mean_score)
    preds = [mean_score for _ in range(len(dev_scores))]
    rmse = compute_rmse(preds, dev_scores)
    print("RMSE:", rmse)
