from abc import ABC, abstractmethod
from typing import Dict

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments

from Experiment import Experiment
from bert.CustomTrainer import CustomTrainer
from bert.hps import hp_tune
from paths import RESULTS_PATH
from utils import get_device, get_hf_args


# Works for all BERT models, but need to be subclasses for each task
class ExperimentBert(Experiment, ABC):

    def __init__(self, task_name: str, model_name: str, data_fraction: float, max_input_length: int, quick_run: bool):
        super().__init__(task_name, model_name, data_fraction)

        # Arguments that are not covered by HPO
        training_arguments = TrainingArguments(
            output_dir=RESULTS_PATH,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=1,
            num_train_epochs=10,
            weight_decay=0.1,
            warmup_ratio=0.06,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            overwrite_output_dir=True,
            skip_memory_metrics=True,
            fp16=True,
            disable_tqdm=True,
            load_best_model_at_end=True,
            report_to=None,
            # report_to=["none"]
        )
        self.training_args = training_arguments
        self.max_input_length = max_input_length
        self.quick_run = quick_run
        if quick_run:
            self.training_args.num_train_epochs = 2
            self.data_fraction = 0.02

    @abstractmethod
    def create_dataset(self, tokenizer, max_seq_len: int):
        pass

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def _load_config(self):
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = self.num_classes
        return config

    def _load_model(self, config, model_path=None):
        model_name = self.model_name if model_path is None else model_path
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        model.to(get_device())
        return model

    def _run_hpo(self, config, tokenizer, train_ds, val_ds):
        def _model_init(trial=None):
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

        trainer = CustomTrainer.init_trainer(self.training_args, None, tokenizer, train_ds, val_ds,
                                             self._compute_metrics, _model_init)
        best_run, best_model_dir = hp_tune(trainer, self.model_name, self.task_name, self.quick_run,
                                           self.metric, self.direction)
        best_model = self._load_model(config, best_model_dir)
        trainer = CustomTrainer.init_trainer(self.training_args, best_model, tokenizer, train_ds, val_ds,
                                             self._compute_metrics)
        return trainer

    def _evaluate(self, trainer, test_ds):
        metrics_eval = trainer.evaluate()
        trainer.log_metrics("eval", metrics_eval)

        predictions, labels, metrics_test = trainer.predict(test_ds)
        # predictions = np.argmax(predictions, axis=1)
        trainer.log_metrics("test", metrics_test)

        metric_dict = {
            "metric": self.metric,
            "eval": metrics_eval["eval_" + self.metric],
            "test": metrics_test["test_" + self.metric]
        }

        return metric_dict

    def run_impl(self) -> Dict[str, float]:
        # model_args, data_args, training_args = get_hf_args()
        # model_name_or_path = model_args.model_name_or_path
        tokenizer = self._load_tokenizer()
        config = self._load_config()
        max_seq_len = min(config.max_position_embeddings, self.max_input_length)

        train_ds, val_ds, test_ds = self.create_dataset(tokenizer, max_seq_len)

        trainer = self._run_hpo(config, tokenizer, train_ds, val_ds)

        metric_dict = self._evaluate(trainer, test_ds)

        # experiment_metric_path = EXPERIMENT_METRICS_PATH_TEMPLATE.format(model=self.model_name, task=self.task_name)
        # with open(experiment_metric_path + "/metrics.json", 'w') as f:
        #     f.write(json.dumps(metric_dict, ensure_ascii=False, indent="\t"))

        return metric_dict
