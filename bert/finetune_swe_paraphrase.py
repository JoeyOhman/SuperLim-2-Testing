import json

import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForSequenceClassification

from bert.CustomTrainer import CustomTrainer
from bert.bert_utils import load_tokenizer, load_config, load_model
from bert.hps import hp_tune
from dataset_loaders.data_loader_swe_paraphrase import load_swe_paraphrase
from paths import EXPERIMENT_MODELS_PATH_TEMPLATE, EXPERIMENT_METRICS_PATH_TEMPLATE
from utils import set_seed, get_device, get_hf_args, task_to_info_dict


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions = np.argmax(predictions, axis=1)
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}
    # return metric.compute(predictions=predictions, references=labels)


def pre_process_data(dataset_split, tokenizer, max_len):
    # Map works sample by sample or in batches if batched=True
    dataset_split = dataset_split.map(
        lambda sample: tokenizer(sample['Sentence 1'], sample['Sentence 2'], truncation=True, max_length=max_len),
        batched=True, num_proc=4)

    # Could use batched=True here and do float conversion in list comprehension
    dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['Score'])}, batched=False, num_proc=4)

    # dataset_split.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset_split.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset_split = dataset_split.remove_columns(['Genre', 'File', 'Sentence 1', 'Sentence 2', 'Score'])
    # dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=batch_size)
    return dataset_split


def create_dataset(data_fraction, tokenizer, max_seq_len):
    train_ds, dev_ds, test_ds = load_swe_paraphrase(data_fraction)

    print("Preprocessing train_ds")
    train_ds = pre_process_data(train_ds, tokenizer, max_seq_len)
    print("Preprocessing dev_ds")
    dev_ds = pre_process_data(dev_ds, tokenizer, max_seq_len)
    print("Preprocessing test_ds")
    test_ds = pre_process_data(test_ds, tokenizer, max_seq_len)

    train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
    print("Train ds, Max len:", max(train_ds_lens))
    print("Train ds, Mean len:", np.mean(train_ds_lens))
    print(f"#samples:\ntrain={train_ds.num_rows}, dev={dev_ds.num_rows}, test={test_ds.num_rows}")

    return train_ds, dev_ds, test_ds


def main():
    model_args, data_args, training_args = get_hf_args()

    model_name_or_path = model_args.model_name_or_path
    tokenizer = load_tokenizer(model_name_or_path)
    config = load_config(model_name_or_path, 1)

    max_seq_len = min(config.max_position_embeddings, data_args.max_input_length)

    train_ds, val_ds, test_ds = create_dataset(data_args.data_fraction, tokenizer, max_seq_len)

    # experiment_model_path = EXPERIMENT_MODELS_PATH_TEMPLATE.format(model=model_name_or_path, task=data_args.task)
    experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=model_name_or_path, task=data_args.task)

    if model_args.hp_search:
        def model_init(trial=None):
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

        trainer = CustomTrainer.init_trainer(training_args, None, tokenizer, train_ds, val_ds, compute_metrics,
                                             model_init)
        task_info = task_to_info_dict[data_args.task]
        best_run, best_model_dir = hp_tune(trainer, model_args.model_name_or_path, data_args.task, data_args.quick_run,
                                           task_info["metric"], task_info["direction"])
        best_model = load_model(best_model_dir, config)
        trainer = CustomTrainer.init_trainer(training_args, best_model, tokenizer, train_ds, val_ds, compute_metrics)

    else:
        model = load_model(model_name_or_path, config)
        trainer = CustomTrainer.init_trainer(training_args, model, tokenizer, train_ds, val_ds, compute_metrics)
        trainer.train()

    metrics_eval = trainer.evaluate()
    trainer.log_metrics("eval", metrics_eval)

    predictions, labels, metrics_test = trainer.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    trainer.log_metrics("test", metrics_test)

    metric_dict = {
        "eval_mse": metrics_eval["eval_rmse"],
        "test_mse": metrics_test["test_rmse"]
    }

    with open(experiment_metric_path + "/metrics.json", 'w') as f:
        f.write(json.dumps(metric_dict, ensure_ascii=False, indent="\t"))


if __name__ == '__main__':
    set_seed()
    device = get_device()
    main()
