from typing import Dict

from transformers import Trainer, DataCollatorWithPadding, EarlyStoppingCallback


class CustomTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        try:
            logs["learning_rate"] = self._get_learning_rate()
        except AttributeError:
            print("Failed to log learning rate.")

        super().log(logs)

    @staticmethod
    def init_trainer(training_args, model, tokenizer, train_ds, val_ds, compute_metrics_fn, model_init=None):
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                                pad_to_multiple_of=8 if training_args.fp16 else None)

        # trainer = Trainer(
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.0)],
            model_init=model_init
        )

        return trainer
