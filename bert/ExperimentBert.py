import torch
import transformers
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments

from Experiment import Experiment
from bert.CustomTrainer import CustomTrainer
from bert.bert_utils import find_best_model_and_clean
from bert.hps import hp_tune
from paths import TRAINER_OUTPUT_PATH
from utils import get_device


# Works for all BERT models, but need to be subclasses for each task
class ExperimentBert(Experiment, ABC):

    def __init__(self, task_name: str, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool,
                 quick_run: bool):
        assert accumulation_steps == 1 or accumulation_steps % 2 == 0, "accumulation_steps must be 1, or multiple of 2"
        super().__init__(task_name, model_name, data_fraction)

        # Arguments that are not covered by HPO
        training_arguments = TrainingArguments(
            learning_rate=1e-4,
            output_dir=TRAINER_OUTPUT_PATH,
            per_device_train_batch_size=int(32 / accumulation_steps),
            per_device_eval_batch_size=int(64 / accumulation_steps),
            gradient_accumulation_steps=accumulation_steps,
            eval_accumulation_steps=accumulation_steps,
            num_train_epochs=10,
            # num_train_epochs=10,
            weight_decay=0.1,
            warmup_ratio=0.06,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            overwrite_output_dir=True,
            skip_memory_metrics=True,
            fp16=torch.cuda.is_available(),
            disable_tqdm=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_" + self.metric,
            greater_is_better=self.direction == "max",
            # load_best_model_at_end=False,
            report_to=None
        )
        self.accumulation_steps = accumulation_steps
        self.training_args = training_arguments
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.max_input_length = max_input_length
        self.hps = hps
        self.quick_run = quick_run
        self.config = self._load_config()
        self.max_seq_len = self.config.max_position_embeddings
        self.tokenizer = self._load_tokenizer()

        if quick_run:
            self.max_seq_len = min(self.max_seq_len, 64)
            self.training_args.num_train_epochs = 2
            self.data_fraction = 0.05

    @staticmethod
    @abstractmethod
    def preprocess_data(dataset_split, tokenizer, max_len: int, dataset_split_name: str):
        pass

    # def _predict(self, model, val_ds, test_ds):
    #     return None

    def create_dataset(self, tokenizer, max_seq_len: int):
        train_ds, dev_ds, test_ds = self._load_data()

        print("Preprocessing train_ds")
        train_ds = self.preprocess_data(train_ds, tokenizer, max_seq_len, "train")
        print("Preprocessing dev_ds")
        dev_ds = self.preprocess_data(dev_ds, tokenizer, max_seq_len, "dev")
        print("Preprocessing test_ds")
        test_ds = self.preprocess_data(test_ds, tokenizer, max_seq_len, "test")

        train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
        print("Train ds, Max len:", max(train_ds_lens))
        print("Train ds, Mean len:", np.mean(train_ds_lens))
        print(f"#samples:\ntrain={train_ds.num_rows}, dev={dev_ds.num_rows}, test={test_ds.num_rows}")

        return train_ds, dev_ds, test_ds

    def _load_tokenizer(self):
        transformers.logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if "gpt" in self.model_name:
            tokenizer.padding_side = "left"
        if "gpt2" in self.model_name or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_config(self):
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = self.num_classes
        if "gpt2" in self.model_name:
            config.pad_token_id = config.eos_token_id
        return config

    def _load_model(self, config, model_path=None):
        model_name = self.model_name if model_path is None else model_path
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config,
                                                                   ignore_mismatched_sizes=True)
        # TODO:
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(self.tokenizer))
        model.to(get_device())
        return model

    def _run_hps(self, config, tokenizer, train_ds, val_ds):
        def _model_init(trial=None):
            # TODO: resize model embedding to match new tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
            model.resize_token_embeddings(len(self.tokenizer))
            return model

        trainer = CustomTrainer.init_trainer(self.training_args, None, tokenizer, train_ds, val_ds,
                                             self._compute_metrics, _model_init)

        best_run = hp_tune(trainer, self.model_name, self.task_name, self.accumulation_steps, self.quick_run,
                           self.metric, self.direction)

        best_model_dir = find_best_model_and_clean(best_run, self.direction, self.task_name, self.model_name)
        best_model = self._load_model(config, best_model_dir)
        trainer = CustomTrainer.init_trainer(self.training_args, best_model, tokenizer, train_ds, val_ds,
                                             self._compute_metrics)
        return trainer

    def _run_no_hps(self, config, tokenizer, train_ds, val_ds):
        model = self._load_model(config)
        trainer = CustomTrainer.init_trainer(self.training_args, model, tokenizer, train_ds, val_ds,
                                             self._compute_metrics)
        trainer.train()
        return trainer

    def _evaluate(self, trainer, val_ds, test_ds):
        # Try overridden predict
        # custom_predict_result = self._predict(trainer.model, val_ds, test_ds)
        # if custom_predict_result is not None:
        #     metrics_eval, metrics_test = custom_predict_result
        #
        # else:
        # Trainer native evaluation
        metrics_eval = trainer.evaluate()
        trainer.log_metrics("eval", metrics_eval)

        predictions, labels, metrics_test = trainer.predict(test_ds)
        # predictions = np.argmax(predictions, axis=1)
        trainer.log_metrics("test", metrics_test)

        # Pack results
        metric_dict = {
            "eval": metrics_eval["eval_" + self.metric],
            "test": metrics_test["test_" + self.metric],
            "eval_metrics_obj": metrics_eval,
            "test_metrics_obj": metrics_test
        }

        return metric_dict

    def run_impl(self) -> Dict[str, float]:
        # model_args, data_args, training_args = get_hf_args()
        # model_name_or_path = model_args.model_name_or_path

        train_ds, val_ds, test_ds = self.create_dataset(self.tokenizer, self.max_seq_len)

        if self.hps:
            trainer = self._run_hps(self.config, self.tokenizer, train_ds, val_ds)
        else:
            trainer = self._run_no_hps(self.config, self.tokenizer, train_ds, val_ds)

        metric_dict = self._evaluate(trainer, val_ds, test_ds)

        # experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=self.model_name, task=self.task_name)
        # with open(experiment_metric_path + "/metrics.json", 'w') as f:
        #     f.write(json.dumps(metric_dict, ensure_ascii=False, indent="\t"))

        return metric_dict
