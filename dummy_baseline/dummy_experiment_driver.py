from dummy_baseline.ExperimentDummy import ExperimentDummy
from dummy_baseline.ExperimentSKLearn import ExperimentSKLearn, SUPPORTED_MODELS
from utils import set_seed


def main():
    tasks = ["SweParaphrase", "ABSAbank-Imm", "SweWiC", "SweFAQ", "SweWinograd", "DaLAJ", "SweMNLI",
             "ArgumentationSentences"]
    # tasks = ["ABSAbank-Imm", "SweWiC", "SweFAQ", "SweWinograd", "DaLAJ", "ArgumentationSentences"]
    # tasks = ["SweMNLI"]
    # tasks = ["SweParaphrase"]
    for task in tasks:
        print(task)
        print("Dummies")
        ExperimentDummy(task, is_random=True).run()
        ExperimentDummy(task, is_random=False).run()
        # if task != "SweFAQ":
        for model_name in SUPPORTED_MODELS:
            print("model:", model_name)
            ExperimentSKLearn(task, model_name).run()


if __name__ == '__main__':
    set_seed()
    main()
