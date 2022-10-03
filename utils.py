from typing import Optional
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments


# Direction (min or max) is whether the optimization should minimize or maximize the metric
task_to_info_dict = {
    "SweParaphrase": {"metric": "rmse", "direction": "min"}
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_hf_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Use quick setup to test if the entire framework works
    if data_args.quick_run:
        training_args.num_train_epochs = 2
        data_args.data_fraction = 0.02
        model_args.model_name_or_path = "albert-base-v2"

    return model_args, data_args, training_args


def debug_print_hf_dataset_sample(hf_dataset, tokenizer):
    for sample in hf_dataset:
        print("First sample:")
        print(sample)

        input_ids = sample['input_ids']
        print("Encoded ids:")
        print(input_ids)

        decoded = tokenizer.decode(input_ids)
        print("Decoded ids:")
        print(decoded)
        break


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    hp_search: Optional[bool] = field(
        default=False, metadata={"help": "Whether to perform grid-search or just train with given parameters"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={
            "help": "the task/dataset to train/evaluate on, e.g. SweParaphrase"
        }
    )
    max_input_length: int = field(
        default=512,
        metadata={
            "help": "The sequence length of samples, longer samples than this value will be truncated."
        },
    )
    data_fraction: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The fraction of the datasets to use, e.g. 0.2 will only use 20% of each dataset split."
        }
    )
    quick_run: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set to true, run with small search-space and a small subset of the data."
        }
    )
