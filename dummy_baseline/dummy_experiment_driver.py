from dummy_baseline.ExperimentDummy import ExperimentDummy
from utils import set_seed


def main():
    tasks = ["SweParaphrase", "SweFAQ", "DaLAJ", "ABSAbank-Imm"]
    for task in tasks:
        ExperimentDummy(task, is_random=True).run()
        ExperimentDummy(task, is_random=False).run()


if __name__ == '__main__':
    set_seed()
    main()
