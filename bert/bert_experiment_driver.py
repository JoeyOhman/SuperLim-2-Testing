from bert.ExperimentBertDaLAJ import ExperimentBertDaLAJ
from bert.ExperimentBertSweFAQ import ExperimentBertSweFAQ
from bert.ExperimentBertSweParaphrase import ExperimentBertSweParaphrase


def main():
    data_fraction = 1.0
    quick_run = False
    hps = False

    ExperimentBertSweFAQ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()
    exit()

    ExperimentBertSweParaphrase("albert-base-v2", data_fraction, hps, quick_run).run()
    ExperimentBertSweParaphrase("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()
    ExperimentBertDaLAJ("albert-base-v2", data_fraction, hps, quick_run).run()
    ExperimentBertDaLAJ("KB/bert-base-swedish-cased", data_fraction, hps, quick_run).run()

    # TODO: make some wrapper func for run that also writes the results to correct place
    # TODO: maybe write everything from there, now hps is writing config I think

    # Add more models + tasks here


if __name__ == '__main__':
    main()
