import json
import os
from collections import defaultdict

from transformers import AutoModel

from paths import ABSOLUTE_PATH_PROJ

MODELS = [
    "KB/bert-base-swedish-cased",
    "KBLab/megatron-bert-base-swedish-cased-600k",
    "KBLab/bert-base-swedish-cased-new",
    "xlm-roberta-base",
    "AI-Nordics/bert-large-swedish-cased",
    "KBLab/megatron-bert-large-swedish-cased-165k",
    "xlm-roberta-large",
    "AI-Sweden-Models/gpt-sw3-126m",
    "AI-Sweden-Models/gpt-sw3-356m",
    "gpt2",
    "gpt2-medium",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "microsoft/mdeberta-v3-base",
    "NbAiLab/nb-bert-base"
]


def load_model_sizes():
    with open(os.path.join(ABSOLUTE_PATH_PROJ, "model_sizes.json"), 'r') as f:
        file_str = f.read()

    return json.loads(file_str)


def main():
    params_dict = defaultdict(int)
    for model_name in MODELS:
        model = AutoModel.from_pretrained(model_name)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name}: {num_params}")
        params_dict[model_name] = num_params

    with open(os.path.join(ABSOLUTE_PATH_PROJ, "model_sizes.json"), 'w') as f:
        json.dump(params_dict, f, ensure_ascii=False, indent="\t")


if __name__ == '__main__':
    main()
