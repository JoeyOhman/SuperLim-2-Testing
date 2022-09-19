import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, HfArgumentParser, \
    DataCollatorWithPadding, TrainingArguments, Trainer, AutoConfig
from datasets import load_dataset, load_metric

from dataset_loaders.data_loader_swe_paraphrase import load_swe_paraphrase
from utils import set_seed, get_device, ModelArguments, DataTrainingArguments


def load_tokenizer_model(model_args):
    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path, do_lower_case=False)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    model.to(device)
    return tokenizer, model
    # return tokenizer, None


def init_trainer(training_args, model, tokenizer, train_ds, val_ds):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    # metric = load_metric('mse')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions = np.argmax(predictions, axis=1)
        rmse = mean_squared_error(labels, predictions, squared=False)
        return {"rmse": rmse}
        # return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return training_args, trainer


def pre_process_data(dataset_split, tokenizer, max_len):
    # Map works sample by sample or in batches if batched=True
    dataset_split = dataset_split.map(
        lambda sample: tokenizer(sample['Sentence 1'], sample['Sentence 2'], truncation=True, max_length=max_len),
        batched=True, num_proc=4)

    # Could use batched=True here and do float conversion in list comprehension
    dataset_split = dataset_split.map(lambda sample: {'labels': float(sample['Score'])}, batched=False, num_proc=4)

    dataset_split.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset_split = dataset_split.remove_columns(['Genre', 'File', 'Sentence 1', 'Sentence 2', 'Score'])
    # dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=BATCH_SIZE)
    return dataset_split


def create_dataset(data_args, tokenizer, max_seq_len):
    train_ds, dev_ds = load_swe_paraphrase(data_args.data_fraction)

    print("Preprocessing train_ds")
    train_ds = pre_process_data(train_ds, tokenizer, max_seq_len)
    print("Preprocessing dev_ds")
    dev_ds = pre_process_data(dev_ds, tokenizer, max_seq_len)

    # print(train_ds)
    train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
    print("Train ds, Max len:", max(train_ds_lens))
    print("Train ds, Mean len:", np.mean(train_ds_lens))

    # for sample in train_ds:
    #     print(sample)
    #     input_ids = sample['input_ids']
    #     print(input_ids)
    #     decoded = tokenizer.decode(input_ids)
    #     print(decoded)
    #     break

    return train_ds, dev_ds


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer, model = load_tokenizer_model(model_args)

    max_seq_len = min(model.config.max_position_embeddings, data_args.max_input_length)

    train_ds, val_ds = create_dataset(data_args, tokenizer, max_seq_len)

    training_args, trainer = init_trainer(training_args, model, tokenizer, train_ds, val_ds)

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # predictions, labels, metrics = trainer.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    # trainer.log_metrics("test", metrics)


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    main()
