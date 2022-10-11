import gc
import time

from bert.ExperimentBertABSAbankImm import ExperimentBertABSAbankImm
from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertSweFAQ import ExperimentBertSweFAQ
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase


def gc_and_wait():
    time.sleep(5)
    gc.collect()
    time.sleep(10)
    gc.collect()
    time.sleep(10)


def main():
    data_fraction = 1.0
    quick_run = False
    hps = True

    # ExperimentBertDaLAJ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 2, data_fraction, hps, quick_run).run()
    # ExperimentBertABSAbankImm("albert-base-v2", data_fraction, hps, quick_run).run()
    # ExperimentBertSweFAQ("xlm-roberta-base", data_fraction, False, quick_run).run()
    # ExperimentBertSweFAQ("KB/bert-base-swedish-cased", data_fraction, False, quick_run).run()
    # exit()
    # ExperimentBertABSAbankImm("AI-Nordics/bert-large-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertSweFAQ("xlm-roberta-large", 2, data_fraction, hps, quick_run).run()
    # gc_and_wait()
    # ExperimentBertABSAbankImm("xlm-roberta-large", 2, data_fraction, hps, quick_run).run()
    # gc_and_wait()
    # ExperimentBertSweParaphrase("xlm-roberta-base", 2, data_fraction, hps, quick_run).run()
    # ExperimentBertSweParaphrase("microsoft/mdeberta-v3-base", 2, data_fraction, hps, quick_run).run()
    # ExperimentBertSweParaphrase("KB/bert-base-swedish-cased", 2, data_fraction, hps, quick_run).run()
    # exit()

    models = [
        # "KB/bert-base-swedish-cased",
        # "KBLab/bert-base-swedish-cased-new",
        # "KBLab/megatron-bert-base-swedish-cased-600k",
        # "KBLab/megatron-bert-large-swedish-cased-165k",
        # "AI-Nordics/bert-large-swedish-cased",
        # "xlm-roberta-base",
        # "xlm-roberta-large",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "microsoft/mdeberta-v3-base",
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    ]
    # "microsoft/mdeberta-v3-base",  # did not work cause of data format either I think?
    # "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" # did not work without customization of data format?
    for model in models:
        ExperimentBertSweParaphrase(model, 1, data_fraction, hps, quick_run).run()
        gc_and_wait()
        ExperimentBertDaLAJ(model, 1, data_fraction, hps, quick_run).run()
        gc_and_wait()
        ExperimentBertSweFAQ(model, 1, data_fraction, hps, quick_run).run()
        gc_and_wait()
        ExperimentBertABSAbankImm(model, 1, data_fraction, hps, quick_run).run()
        gc_and_wait()

    # ExperimentBertSweFAQ("AI-Nordics/bert-large-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertSweFAQ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()

    # ExperimentBertSweParaphrase("albert-base-v2", data_fraction, hps, quick_run).run()
    # ExperimentBertSweParaphrase("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()
    # ExperimentBertDaLAJ("albert-base-v2", data_fraction, hps, quick_run).run()
    # ExperimentBertDaLAJ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()


if __name__ == '__main__':
    main()
