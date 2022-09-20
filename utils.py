from typing import Optional
import torch
import random
import numpy as np
from dataclasses import dataclass, field


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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
    max_input_length: int = field(
        default=512,
        metadata={
            "help": "The sequence length of samples, longer samples than this value will be truncated."
        },
    )
    data_fraction: float = field(
        default=1.0,
        metadata={
            "help": "The fraction of the datasets to use, e.g. 0.2 will only use 20% of each dataset split."
        }
    )
