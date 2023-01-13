# SuperLim-2-Testing

This repository implements experiments/baselines for SuperLim 2.

Repository development guide:

The class hierarchy handles much of the execution flow, like a framework.

## Utils

- `paths.py` defines paths that can be imported, and should work cross-platform and in docker containers etc 
- `compute_metrics.py` implements functions to compute metrics required for the experiments
- `collect_metrics.py` loads the experiment results in the pre-defined results file hierarchy and plots a table. TODO: make this generate json-files according to template.

## Class Hierarchy

###  Experiment.py
`Experiment.py` defines the abstract class `Experiment`, which handles the 
experiment execution flow, keeps track of the model name, task name, and other meta-data. 
It also handles the writing of results to file, and contains `task_to_info_dict` which 
should contain the required meta-data for each task, such as which metric is used. 

### ExperimentBert Class
`ExperimentBert.py` defines another abstract class `ExperimentBert`, which handles 
loading of tokenizers and models, default hyperparameters, hyperparameter tuning, 
HuggingFace Trainer, Evaluation, etc. 

### ExperimentBert Children
Children of `ExperimentBert` simply define the task name, which should be supported in `Experiment.py`, 
and implements the abstract method `create_dataset`, which calls the `_load_data` method 
and preprocesses it to make it ready for training. 

### ExperimentDummy
Inherits Experiment, and uses `sklearn.dummy` to provide a dummy baseline. 
This works for any task that has training data and is a classification or regression problem.

## Adding a new task
1. Make sure the data is accessible, preferably via HF Datasets, or otherwise through TSV files
2. Create a function `reformat_dataset_<task_name>` in `dataset_loader.py`, that converts the dataset into a dataset of desired format.
3. Add a pointer to this function in the dictionary `TASK_TO_REFORMAT_FUN` above the `load_dataset_by_task` function in the same file. 
4. Add an entry with the required meta-data for this task in `Experiment.py`, in the dictionary `task_to_info_dict`.
5. Add an entry in `bert_experiment_driver.py` for the corresponding ExperimentBert class
6. The dataset, reformatting, and meta-data loading is done automatically when a child instance of `Experiment` is used.

## Results structure
- Metric results are stored in json-files with the following path:
`results/experiments/metrics/<task_name>/<model_name>/metrics.json`

- Ray (library for HPS) results are stored in `results/ray_results`, and can be cleared
after the experiments are done. Can be done automatically with `clear_result_checkpoints.sh`.

- HF Trainer checkpoints are stored in `results/trainer_output`, and can be cleared 
after the experiments are done. This is done automatically in the `BertExperiment` class, and with `clear_result_checkpoints.sh`

