import json
import os

import torch
import transformers
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments

from Experiment import Experiment
from bert.CustomTrainer import CustomTrainer
from bert.bert_utils import find_best_model_and_clean
from bert.hps import hp_tune
from paths import TRAINER_OUTPUT_PATH, MODELS_PATH, get_experiment_metrics_path
from utils import get_device


# Works for all TRANSFORMER models, including GPT, but need to be subclassed for each task
class ExperimentBert(Experiment, ABC):

    def __init__(self, task_name: str, model_name: str, accumulation_steps: int, data_fraction: float, hps: bool,
                 quick_run: bool, evaluate_only: bool):
        assert accumulation_steps == 1 or accumulation_steps % 2 == 0, "accumulation_steps must be 1, or multiple of 2"
        if evaluate_only:
            safe_task = task_name.replace("/", "-")
            safe_model = model_name.replace("/", "-")
            self.model_path = os.path.join(MODELS_PATH, safe_task, safe_model)
        else:
            self.model_path = model_name
        super().__init__(task_name, model_name, data_fraction, evaluate_only)

        self.num_train_epochs = 10
        # self.num_train_epochs = 1
        self.warmup_ratio = 0.06
        self.weight_decay = 0.0 if "gpt" in self.model_name else 0.1  # RoBERTa GLUE = 0.1, let GPT use HF default = 0.0
        self.fp16 = torch.cuda.is_available()
        # Arguments that are not covered by HPO
        training_arguments = TrainingArguments(
            # learning_rate=1e-4,
            output_dir=TRAINER_OUTPUT_PATH,
            per_device_train_batch_size=int(32 / accumulation_steps),
            per_device_eval_batch_size=int(64 / accumulation_steps),
            # per_device_train_batch_size=1,
            # per_device_eval_batch_size=1,
            gradient_accumulation_steps=accumulation_steps,
            eval_accumulation_steps=accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            overwrite_output_dir=True,
            skip_memory_metrics=True,
            fp16=self.fp16,
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
        self.use_resize = True
        self.hps = hps
        self.quick_run = quick_run
        self.config = self._load_config()
        self.max_seq_len = self.config.max_position_embeddings
        self.tokenizer = self._load_tokenizer()

        # Avoid technical issues by not going beyond 1024
        # self.max_seq_len = min(self.max_seq_len, 1024)
        # self.max_seq_len = min(self.max_seq_len, 512)
        # self.max_seq_len = min(self.max_seq_len, 256)
        self.max_seq_len = min(self.max_seq_len, 128)
        # self.max_seq_len = min(self.max_seq_len, 80)
        # self.max_seq_len = min(self.max_seq_len, 64)
        if quick_run:
            self.max_seq_len = min(self.max_seq_len, 64)
            self.training_args.num_train_epochs = 2
            self.data_fraction = 0.05

    @abstractmethod
    def preprocess_data(self, dataset_split):
        pass

    # def _predict(self, model, val_ds, test_ds):
    #     return None

    def create_dataset(self):
        train_ds, dev_ds, test_ds = self._load_data()

        print("Preprocessing train_ds")
        train_ds = self.preprocess_data(train_ds)
        print("Preprocessing dev_ds")
        dev_ds = self.preprocess_data(dev_ds)
        print("Preprocessing test_ds")
        test_ds = self.preprocess_data(test_ds)

        train_ds_lens = [sample['input_ids'].shape[0] for sample in train_ds]
        print("Train ds, Max len:", max(train_ds_lens))
        print("Train ds, Mean len:", np.mean(train_ds_lens))
        print(f"#samples:\ntrain={train_ds.num_rows}, dev={dev_ds.num_rows}, test={test_ds.num_rows}")

        return train_ds, dev_ds, test_ds

    def tokenize_gpt(self, texts1, texts2=None, texts3=None, texts4=None, ret_tensors=None):
        sep = " $ "
        # Only texts1 (2, 3, 4 = None)
        if texts2 is None and texts3 is None and texts4 is None:
            return self.tokenizer(texts1,
                                  truncation=True, max_length=self.max_seq_len, padding=True,
                                  return_tensors=ret_tensors)

        # Only texts1 and texts2 (3, 4 = None)
        elif texts3 is None and texts4 is None:
            return self.tokenizer(
                [t1 + sep + t2 for t1, t2 in zip(texts1, texts2)],
                truncation=True, max_length=self.max_seq_len, padding=True, return_tensors=ret_tensors)

        # Only texts1, texts2, and texts3 (4 = None)
        elif texts4 is None:
            return self.tokenizer(
                [t1 + sep + t2 + sep + t3 for t1, t2, t3 in zip(texts1, texts2, texts3)],
                truncation=True, max_length=self.max_seq_len, padding=True, return_tensors=ret_tensors)
        # All 4 texts
        else:
            return self.tokenizer(
                [t1 + sep + t2 + sep + t3 + sep + t4 for t1, t2, t3, t4 in zip(texts1, texts2, texts3, texts4)],
                truncation=True, max_length=self.max_seq_len, padding=True, return_tensors=ret_tensors)

    def tokenize_bert(self, texts1, texts2=None, texts3=None, texts4=None, ret_tensors=None):

        # Combine texts3 and texts4
        if texts4 is not None:
            texts3 = [t3 + " [SEP] " + t4 for t3, t4 in zip(texts3, texts4)]
            texts4 = None

        # Combine texts2 and texts3
        if texts3 is not None:
            texts2 = [t2 + " [SEP] " + t3 for t2, t3 in zip(texts2, texts3)]
            texts3 = None

        # Only texts1
        if texts2 is None:
            # single text
            return self.tokenizer(texts1, truncation=True, max_length=self.max_seq_len, padding=True,
                                  return_tensors=ret_tensors)

        # Only texts1 and texts2
        else:
            # text-pair
            return self.tokenizer(texts1, texts2, truncation=True, max_length=self.max_seq_len, padding=True,
                                  return_tensors=ret_tensors)

        # All 3 texts
        # else:
        #     print("All 3 texts")
        #     return self.tokenizer(texts1, texts2, texts3, truncation=True, max_length=self.max_seq_len, padding=True,
        #                           return_tensors=ret_tensors)

    def batch_tokenize(self, texts1, texts2=None, texts3=None, texts4=None, ret_tensors=None):
        # Don't allow strings directly, only batches (even if size 1 batch)
        assert not isinstance(texts1, str)
        # Don't allow sending 2 texts, with texts1 and texts3
        if texts3 is None:
            assert texts4 is None
        if texts2 is None:
            assert texts3 is None
            assert texts4 is None

        if "gpt" in self.model_name:
            return self.tokenize_gpt(texts1, texts2, texts3, texts4, ret_tensors)
        else:
            return self.tokenize_bert(texts1, texts2, texts3, texts4, ret_tensors)

    def _load_tokenizer(self):
        transformers.logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if "gpt" in self.model_name:
            tokenizer.padding_side = "left"
            if self.use_resize:
                tokenizer.add_tokens(['$'], special_tokens=True)
            # tokenizer.add_tokens(['$', '[CLS]'], special_tokens=True)
        if "gpt2" in self.model_name or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_config(self):
        config = AutoConfig.from_pretrained(self.model_path)
        config.num_labels = self.num_classes
        if "gpt2" in self.model_name or config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id
        return config

    def _load_model(self, config, model_path=None):
        model_name = self.model_path if model_path is None else model_path
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config,
                                                                   ignore_mismatched_sizes=True)

        if "gpt" in self.model_name and self.use_resize:
            model.resize_token_embeddings(len(self.tokenizer))
        model.to(get_device())
        return model

    def _run_hps(self, config, tokenizer, train_ds, val_ds):
        def _model_init(trial=None):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path, config=config,
                                                                       ignore_mismatched_sizes=True)
            if "gpt" in self.model_name and self.use_resize:
                model.resize_token_embeddings(len(self.tokenizer))
            return model

        trainer = CustomTrainer.init_trainer(self.training_args, None, tokenizer, train_ds, val_ds,
                                             self._compute_metrics, _model_init)

        best_run = hp_tune(trainer, self.model_name, self.task_name, self.accumulation_steps, self.quick_run,
                           self.metric, self.direction)

        best_model_dir, hp_dict = find_best_model_and_clean(best_run, self.direction, self.task_name, self.model_name)
        best_model = self._load_model(config, best_model_dir)
        trainer = CustomTrainer.init_trainer(self.training_args, best_model, tokenizer, train_ds, val_ds,
                                             self._compute_metrics)
        return trainer, hp_dict

    def _run_no_hps(self, config, tokenizer, train_ds, val_ds):
        model = self._load_model(config)
        trainer = CustomTrainer.init_trainer(self.training_args, model, tokenizer, train_ds, val_ds,
                                             self._compute_metrics)
        trainer.train()
        hp_dict = {
            "learning_rate": trainer.args.learning_rate,
            "per_device_train_batch_size": trainer.args.per_device_train_batch_size
        }
        return trainer, hp_dict

    def _load_metrics_hp_dict(self):
        metrics_path = get_experiment_metrics_path(self.task_name, self.model_name)
        metrics_file = os.path.join(metrics_path, "metrics.json")
        with open(metrics_file, 'r') as f:
            metrics_dict = json.loads(f.read())

        return metrics_dict["hyperparameters"]

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
        predictions_eval, labels_eval, metrics_eval_not_used = trainer.predict(val_ds)

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

        return metric_dict, predictions_eval, predictions

    def run_impl(self) -> Tuple[Dict[str, float], list, list]:
        # model_args, data_args, training_args = get_hf_args()
        # model_name_or_path = model_args.model_name_or_path

        # TODO: make sure no task alters the convention of one dataset row (differently than raw dataset)
        train_ds, val_ds, test_ds = self.create_dataset()

        if self.evaluate_only:
            print("Only evaluating, no training will be done!")
            # TODO:
            #  Load trained model
            #  Get it into Trainer (maybe a model can be loaded with Trainer)
            #  Call evaluate like below, maybe just if-statement
            model = self._load_model(self.config)
            trainer = CustomTrainer.init_trainer(self.training_args, model, self.tokenizer, train_ds, val_ds,
                                                 self._compute_metrics)
            hp_dict = self._load_metrics_hp_dict()
            pass

        else:
            if self.hps:
                trainer, hp_dict = self._run_hps(self.config, self.tokenizer, train_ds, val_ds)
            else:
                trainer, hp_dict = self._run_no_hps(self.config, self.tokenizer, train_ds, val_ds)

            hp_dict["batch_size"] = hp_dict["per_device_train_batch_size"] * self.accumulation_steps
            del hp_dict["per_device_train_batch_size"]

            hp_dict["num_train_epochs"] = self.num_train_epochs
            hp_dict["warmup_ratio"] = self.warmup_ratio
            hp_dict["weight_decay"] = self.weight_decay
            hp_dict["fp16"] = self.fp16

        # Let ExperimentBertSweMNLI use this
        self.hp_dict = hp_dict

        metric_dict, predictions_eval, predictions_test = self._evaluate(trainer, val_ds, test_ds)

        metric_dict["hyperparameters"] = hp_dict

        # experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=self.model_name, task=self.task_name)
        # with open(experiment_metric_path + "/metrics.json", 'w') as f:
        #     f.write(json.dumps(metric_dict, ensure_ascii=False, indent="\t"))

        return metric_dict, predictions_eval, predictions_test
