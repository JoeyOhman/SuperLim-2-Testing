from dummy_baseline.ExperimentDummy import ExperimentDummy


def main():
    tasks = ["SweParaphrase", "SweFAQ", "DaLAJ", "ABSAbank-Imm"]
    for task in tasks:
        ExperimentDummy(task).run()


if __name__ == '__main__':
    main()
