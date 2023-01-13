import numpy as np
import torch
import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, HfArgumentParser, \
    DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error

from utils import set_seed, get_device

# MODEL_NAME = "bert-base-cased"
# MODEL_NAME = "gpt2"
# MODEL_NAME = "AI-Sweden/gpt-sw3-126m-private"
# MODEL_NAME = "KB/bert-base-swedish-cased"
MODEL_NAME = "microsoft/mdeberta-v3-base"
DATASET_NAME = "swedish_reviews"
# DATASET_NAME = "stsb_mt_sv"
DF = 0.10
# DF = 1.0

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"


def load_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 2 if DATASET_NAME == "swedish_reviews" else 1
    if "gpt" in MODEL_NAME:
        # tokenizer.padding_side = "left"
        tokenizer.add_tokens(['$'], special_tokens=True)
        # tokenizer.add_tokens(['$', '[CLS]'], special_tokens=True)
    if "gpt2" in MODEL_NAME or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


def pre_process_data(tokenizer, dataset_split, max_len):
    if DATASET_NAME == "stsb_mt_sv":
        if "gpt" in MODEL_NAME:
            dataset_split = dataset_split.map(
                lambda sample: tokenizer([t1 + " " + "$" + " " + t2 for t1, t2 in zip(sample['sentence1'], sample['sentence2'])],
                                         truncation=True, max_length=max_len, padding=True, return_tensors="pt"),
                batched=True, num_proc=4)
        else:
            dataset_split = dataset_split.map(
                lambda sample: tokenizer(sample['sentence1'], sample['sentence2'], truncation=True, max_length=max_len,
                                         padding=True, return_tensors="pt"),
                batched=True, num_proc=4)

        dataset_split = dataset_split.map(lambda sample: {'labels': sample['score']}, batched=True, num_proc=4)
    else:
        dataset_split = dataset_split.map(
            lambda sample: tokenizer(sample['text'], truncation=True, max_length=max_len, padding=True, return_tensors="pt"),
            # lambda sample: tokenizer([t + " [CLS]" for t in sample['text']], truncation=True, max_length=max_len, padding=True),
            batched=True, num_proc=4)

        dataset_split = dataset_split.map(lambda sample: {'labels': sample['label']}, batched=True, num_proc=4)

    cols = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    if "gpt" in MODEL_NAME:
        cols.remove("token_type_ids")
    dataset_split.set_format(type='torch', columns=cols)

    if DATASET_NAME == "stsb_mt_sv":
        dataset_split = dataset_split.remove_columns(['sentence1', 'sentence2', 'score'])
    else:
        dataset_split = dataset_split.remove_columns(['text', 'label'])
    # dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=BATCH_SIZE)
    return dataset_split


def create_sentiment_analysis_dataset(tokenizer, max_len):
    print("Loading and pre-processing dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']

    assert 0.0 < DF <= 1.0, "data_fraction must be in range (0, 1]"
    if DF < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * DF)))
        val_ds = val_ds.select(range(int(len(val_ds) * DF)))
        test_ds = test_ds.select(range(int(len(test_ds) * DF)))

    train_ds = pre_process_data(tokenizer, train_ds, max_len)
    val_ds = pre_process_data(tokenizer, val_ds, max_len)
    test_ds = pre_process_data(tokenizer, test_ds, max_len)

    train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
    print("Train ds, Max len:", max(train_ds_lens))
    print("Train ds, Mean len:", np.mean(train_ds_lens))

    return train_ds, val_ds, test_ds


def init_trainer(model, tokenizer, train_ds, val_ds):

    fp16 = torch.cuda.is_available()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if fp16 else None)

    if DATASET_NAME == "swedish_reviews":
        metric = load_metric('accuracy')
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
    else:
        def compute_metrics(preds_labels_tuple):
            predictions, labels = preds_labels_tuple
            # predictions = np.argmax(predictions, axis=1)
            rmse = mean_squared_error(labels, predictions, squared=False)
            return {"rmse": rmse}

    accumulation_steps = 2
    training_arguments = TrainingArguments(
        # learning_rate=1e-4,
        learning_rate=1e-5,
        output_dir="_finetune_bert_output",
        per_device_train_batch_size=int(32 / accumulation_steps),
        per_device_eval_batch_size=int(64 / accumulation_steps),
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation_steps,
        eval_accumulation_steps=accumulation_steps,
        # num_train_epochs=10,
        num_train_epochs=2,
        weight_decay=0.0 if "gpt" in MODEL_NAME else 0.1,  # RoBERTa GLUE = 0.1, let GPT use HF default = 0.0
        # warmup_ratio=0.06,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        overwrite_output_dir=True,
        skip_memory_metrics=True,
        fp16=fp16,
        disable_tqdm=True,
        # load_best_model_at_end=False,
        # metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        # load_best_model_at_end=False,
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return training_arguments, trainer


def main():
    tokenizer, model = load_tokenizer_model()
    max_seq_len = min(model.config.max_position_embeddings, 128)
    train_ds, val_ds, test_ds = create_sentiment_analysis_dataset(tokenizer, max_seq_len)
    training_args, trainer = init_trainer(model, tokenizer, train_ds, val_ds)

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    predictions, labels, metrics = trainer.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    trainer.log_metrics("test", metrics)


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    main()
