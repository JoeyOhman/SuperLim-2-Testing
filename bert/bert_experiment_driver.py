from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase


def main():
    data_fraction = 1.0
    quick_run = False

    # ExperimentBertSweParaphrase("albert-base-v2", data_fraction, quick_run).run()
    # ExperimentBertSweParaphrase("KB/bert-base-swedish-cased", data_fraction, quick_run).run()
    ExperimentBertDaLAJ("albert-base-v2", data_fraction, quick_run).run()

    # TODO: make some wrapper func for run that also writes the results to correct place
    # TODO: maybe write everything from there, now hps is writing config I think

    # Add more models + tasks here


if __name__ == '__main__':
    main()
