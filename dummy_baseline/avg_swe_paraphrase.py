import numpy as np

from sklearn.metrics import mean_squared_error

from dataset_loaders.data_loader_swe_paraphrase import load_swe_paraphrase


def compute_rmse(predictions, labels):
    return mean_squared_error(labels, predictions, squared=False)


if __name__ == '__main__':
    train_ds, dev_ds, test_ds = load_swe_paraphrase()
    train_scores = [float(sample['Score']) for sample in train_ds]
    test_scores = [float(sample['Score']) for sample in test_ds]
    mean_score = np.mean(train_scores)
    # print(mean_score)
    preds = [mean_score for _ in range(len(test_scores))]
    rmse = compute_rmse(preds, test_scores)
    print("RMSE:", rmse)
