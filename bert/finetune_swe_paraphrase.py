import os

import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import HfArgumentParser, TrainingArguments, AutoModelForSequenceClassification, AutoConfig, \
    AutoTokenizer

from bert.CustomTrainer import CustomTrainer
from dataset_loaders.data_loader_swe_paraphrase import load_swe_paraphrase
from utils import set_seed, get_device, ModelArguments, DataTrainingArguments

# os.environ["WANDB_DISABLED"] = "true"

from bert.hps import hp_tune


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_config(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = 1
    return config


def load_model(model_name_or_path, config):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.to(device)
    return model


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


def create_dataset(data_args, tokenizer, max_seq_len):
    train_ds, dev_ds = load_swe_paraphrase(data_args.data_fraction)

    print("Preprocessing train_ds")
    train_ds = pre_process_data(train_ds, tokenizer, max_seq_len)
    print("Preprocessing dev_ds")
    dev_ds = pre_process_data(dev_ds, tokenizer, max_seq_len)

    train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
    print("Train ds, Max len:", max(train_ds_lens))
    print("Train ds, Mean len:", np.mean(train_ds_lens))

    return train_ds, dev_ds


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_name_or_path = model_args.model_name_or_path
    tokenizer = load_tokenizer(model_name_or_path)
    config = load_config(model_name_or_path)

    max_seq_len = min(config.max_position_embeddings, data_args.max_input_length)

    train_ds, val_ds = create_dataset(data_args, tokenizer, max_seq_len)

    if model_args.hp_search:
        def model_init(trial=None):
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

        trainer = CustomTrainer.init_trainer(training_args, None, tokenizer, train_ds, val_ds, compute_metrics,
                                             model_init)
        best_run, best_model_dir = hp_tune(trainer, model_args.model_name_or_path)
        best_model = load_model(best_model_dir, config)
        trainer = CustomTrainer.init_trainer(training_args, best_model, tokenizer, train_ds, val_ds, compute_metrics)

    else:
        model = load_model(model_name_or_path, config)
        trainer = CustomTrainer.init_trainer(training_args, model, tokenizer, train_ds, val_ds, compute_metrics)
        trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # predictions, labels, metrics = trainer.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    # trainer.log_metrics("test", metrics)


if __name__ == '__main__':
    set_seed()
    device = get_device()
    main()
