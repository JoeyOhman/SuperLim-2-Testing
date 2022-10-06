import gc
import time

from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertSweFAQ import ExperimentBertSweFAQ
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase


def main():
    data_fraction = 1.0
    quick_run = False
    hps = True

    # ExperimentBertDaLAJ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertSweParaphrase("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", data_fraction, hps, quick_run).run()
    # ExperimentBertSweFAQ("xlm-roberta-base", data_fraction, False, quick_run).run()
    # ExperimentBertSweFAQ("KB/bert-base-swedish-cased", data_fraction, False, quick_run).run()
    # exit()

    models = [
        # "KB/bert-base-swedish-cased",
        # "KBLab/bert-base-swedish-cased-new",
        # "KBLab/megatron-bert-base-swedish-cased-600k",
        # "KBLab/megatron-bert-large-swedish-cased-165k",
        # "AI-Nordics/bert-large-swedish-cased",
        # "xlm-roberta-base",
        "xlm-roberta-large",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "microsoft/mdeberta-v3-base",
    ]
    # "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" # did not work without customization of data format?
    for model in models:
        ExperimentBertSweParaphrase(model, data_fraction, hps, quick_run).run()
        gc.collect()
        time.sleep(5)
        ExperimentBertDaLAJ(model, data_fraction, hps, quick_run).run()
        gc.collect()
        time.sleep(5)
        ExperimentBertSweFAQ(model, data_fraction, hps, quick_run).run()
        gc.collect()
        time.sleep(5)

    # ExperimentBertSweFAQ("AI-Nordics/bert-large-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertSweFAQ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()

    # ExperimentBertSweParaphrase("albert-base-v2", data_fraction, hps, quick_run).run()
    # ExperimentBertSweParaphrase("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertDaLAJ("albert-base-v2", data_fraction, hps, quick_run).run()
    # ExperimentBertDaLAJ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()


if __name__ == '__main__':
    main()
