import argparse

import transformers

from bert.ExperimentBertABSAbankImm import ExperimentBertABSAbankImm
from bert.ExperimentBertArgumentationSentences import ExperimentBertArgumentationSentences
from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertReviews import ExperimentBertReviews
from bert.ExperimentBertSweFAQ import ExperimentBertSweFAQ
from bert.ExperimentBertSweMNLI import ExperimentBertSweMNLI
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase
from bert.ExperimentBertSweWiC import ExperimentBertSweWiC
from bert.ExperimentBertSweWinograd import ExperimentBertSweWinograd
from utils import set_seed

task_to_bert_class = {
    "ArgumentationSentences": ExperimentBertArgumentationSentences,
    "ABSAbank-Imm": ExperimentBertABSAbankImm,
    "DaLAJ": ExperimentBertDaLAJ,
    "SweFAQ": ExperimentBertSweFAQ,
    "SweParaphrase": ExperimentBertSweParaphrase,
    "SweWiC": ExperimentBertSweWiC,
    "SweWinograd": ExperimentBertSweWinograd,
    "SweMNLI": ExperimentBertSweMNLI,
    "Reviews": ExperimentBertReviews,
}


ACC_STEPS_2 = []
# ACC_STEPS_2 = ["AI-Nordics/bert-large-swedish-cased", "KBLab/megatron-bert-large-swedish-cased-165k", "xlm-roberta-base"]
ACC_STEPS_4 = ["AI-Sweden-Models/gpt-sw3-126m", "gpt2", "xlm-roberta-large"]
ACC_STEPS_8 = ["AI-Sweden-Models/gpt-sw3-356m", "gpt2-medium"]


def main(args):
    data_fraction = 1.0
    # data_fraction = 0.01
    # data_fraction = 0.25
    quick_run = False
    hps = True

    if args.model_name in ACC_STEPS_8:
        accumulation_steps = 8
    elif args.model_name in ACC_STEPS_4:
        accumulation_steps = 4
    elif args.model_name in ACC_STEPS_2:
        accumulation_steps = 2
    else:
        accumulation_steps = 1

    # accumulation_steps = 4

    bert_class = task_to_bert_class[args.task_name]
    bert_class(args.model_name, accumulation_steps, data_fraction, hps, quick_run, args.evaluate_only).run()


if __name__ == '__main__':
    set_seed()
    transformers.logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g. bert-base-cased")
    parser.add_argument("--task_name", type=str, required=True, help="e.g. SweParaphrase")
    parser.add_argument("--evaluate_only", action='store_true', help="Evaluate only, i.e. no training")
    parser.set_defaults(evaluate_only=False)
    _args = parser.parse_args()
    main(_args)
