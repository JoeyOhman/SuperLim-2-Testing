from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from utils import get_device


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_config(model_name_or_path, num_labels):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    return config


def load_model(model_name_or_path, config):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.to(get_device())
    return model