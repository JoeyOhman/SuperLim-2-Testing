import argparse

import transformers

from bert.ExperimentBertABSAbankImm import ExperimentBertABSAbankImm
from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertSweFAQ import ExperimentBertSweFAQ
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase
from utils import set_seed

task_to_bert_class = {
    "ABSAbankImm": ExperimentBertABSAbankImm,
    "DaLAJ": ExperimentBertDaLAJ,
    "SweFAQ": ExperimentBertSweFAQ,
    "SweParaphrase": ExperimentBertSweParaphrase
}


def main(args):
    data_fraction = 1.0
    quick_run = False
    hps = False
    accumulation_steps = 32

    bert_class = task_to_bert_class[args.task_name]
    bert_class(args.model_name, accumulation_steps, data_fraction, hps, quick_run).run()


if __name__ == '__main__':
    set_seed()
    transformers.logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g. bert-base-cased")
    parser.add_argument("--task_name", type=str, required=True, help="e.g. SweParaphrase")
    _args = parser.parse_args()
    main(_args)
