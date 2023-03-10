# SuperLim-2-Testing

This repository implements experiments/baselines for SuperLim 2.

## Reproduce results

### Setup Environment (optional)

```
1. wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
2. bash Anaconda3-2022.10-Linux-x86_64.sh
3. yes, yes, yes
4. conda create -n ptgpu_venv python=3.9
5. conda activate ptgpu_venv
6. conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Setup Repository (optional)

```
1. git clone git@github.com:JoeyOhman/SuperLim-2-Testing.git
2. cd SuperLim-2-Testing
3. pip install -r requirements.txt
4. tmux new -s exp
5. tmux attach -t exp
```

### Run Experiments

```
1. Download all data to data directory 
2. Setup wandb key in api_wandb_key.txt
3. Configure GPUs to use in run_bert_experiments.sh
4. Specify models and tasks in run_bert_experiments.sh
5. Specify accumulation sizes for models in bert/bert_experiment_driver.py to suit your available GPU memory.
6. ./run_dummy_experiments.sh
7. ./run_bert_experiments.sh (this includes gpt experiments if wanted)
8. Run the collect_results/create_table.py script, it will collect results in results/experiments/metrics directory and create the results/experiments/metrics/model_deliverables directory with json files. 
```

This will fine-tune 672 models in total, if the 4 gpt-models are included (they are not by default). 
84 models will be saved, one for each task-model combination.

### Find The Results

```
1. Individual experiment results: results/experiments/metrics/
2. Packed results grouped by model: results/experiments/metrics/model_deliverables/
3. Model predictions on dev and test files: results/experiments/predictions/
4. Best fine-tuned models: results/experiments/models/
```

The tasks are named slightly different in this repository since it was developed during the development of SuperLim 2.
See `task_to_info_dict` in `Experiment.py` for the task names used. When the final results files are created, these 
names are mapped to the official names.


<br>

## **Repository development guide**

The class hierarchy handles much of the execution flow, like a framework.

## Utils

- `paths.py` defines paths that can be imported, and should work cross-platform and in docker containers etc 
- `compute_metrics.py` implements functions to compute metrics required for the experiments
- `collect_metrics.py` loads the experiment results in the pre-defined results file hierarchy and plots a table. 

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
1. Make sure the data is accessible, preferably via HF Datasets, or otherwise through JSONL/TSV files
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


## Results

All tasks are evaluated using the Krippendorff Alpha metric (nominal for classification, and interval for regression).

### Evaluation Results
| Model                                        |       ABSA |     DaLAJ |     SweMNLI |   SweParaphrase |     SweWiC |   SweWinograd |   Swedish FAQ |      Avg ↑ |
|----------------------------------------------|------------|-----------|-------------|-----------------|------------|---------------|---------------|------------|
| KBLab/megatron-bert-large-swedish-cased-165k |  0.57283   |  0.727593 |  0.776482   |       0.909077  |  0.308667  |    0.258708   |    0.892967   |  0.635189  |
| AI-Nordics/bert-large-swedish-cased          |  0.534861  |  0.716007 |  0.746819   |       0.895624  |  0.285408  |    0.240471   |    0.864126   |  0.611902  |
| KB/bert-base-swedish-cased                   |  0.52187   |  0.715983 |  0.720911   |       0.900723  |  0.408507  |    0.188203   |    0.737305   |  0.599072  |
| KBLab/megatron-bert-base-swedish-cased-600k  |  0.503377  |  0.690426 |  0.744255   |       0.895426  |  0.291889  |    0.179137   |    0.873585   |  0.59687   |
| xlm-roberta-large                            |  0.558885  |  0.71126  |  0.783054   |       0.908835  |  0.34125   |    0.151711   |    0.6791     |  0.590585  |
| NbAiLab/nb-bert-base                         |  0.456362  |  0.638692 |  0.717407   |       0.879765  |  0.351611  |    0.177085   |    0.707324   |  0.561178  |
| KBLab/bert-base-swedish-cased-new            |  0.477748  |  0.726241 |  0.727923   |       0.812842  |  0.225232  |    0.0651066  |    0.610978   |  0.520867  |
| xlm-roberta-base                             |  0.402649  |  0.671511 |  0.720407   |       0.871642  |  0.253254  |   -0.251163   |    0.619978   |  0.469754  |
| SVM                                          |  0.336347  |  0.501149 |  0.126261   |       0.173238  |  0.012941  |    0.0981801  |    0.0282215  |  0.182334  |
| Decision Tree                                |  0.210584  |  0.294933 |  0.143506   |       0.248115  |  0.0405762 |    0.0307165  |    0.0185908  |  0.141003  |
| Random                                       | -0.0566656 |  0.022797 |  0.00267875 |      -0.0506558 |  0.0568082 |   -0.00417216 |   -0.0959506  | -0.01788   |
| Random Forest                                |  0.0120151 | -0.31043  | -0.255086   |       0.160169  |  0.0272561 |   -0.251163   |    0.00856598 | -0.0869533 |
| MaxFreq/Avg                                  | -0.0309908 | -0.347135 | -0.337699   |      -0.0242883 | -0.332     |   -0.251163   |   -0.316185   | -0.234209  |

### Test Results
| Model                                        |        ABSA |       DaLAJ |      SweMNLI |   SweParaphrase |      SweWiC |   SweWinograd |   Swedish FAQ |      Avg ↑ |
|----------------------------------------------|-------------|-------------|--------------|-----------------|-------------|---------------|---------------|------------|
| KBLab/megatron-bert-large-swedish-cased-165k |  0.509004   |  0.753261   |  0.231612    |      0.873908   |  0.30598    |     0.188953  |     0.796635  |  0.522765  |
| AI-Nordics/bert-large-swedish-cased          |  0.481386   |  0.745449   |  0.240594    |      0.862353   |  0.316298   |     0.191522  |     0.709803  |  0.506772  |
| KB/bert-base-swedish-cased                   |  0.521762   |  0.739715   |  0.179116    |      0.844902   |  0.37619    |     0.139458  |     0.622103  |  0.489035  |
| xlm-roberta-large                            |  0.518696   |  0.737508   |  0.20472     |      0.881723   |  0.369683   |     0.0806007 |     0.563868  |  0.479543  |
| KBLab/megatron-bert-base-swedish-cased-600k  |  0.451832   |  0.718029   |  0.217683    |      0.866851   |  0.28297    |     0.0614488 |     0.748954  |  0.478253  |
| NbAiLab/nb-bert-base                         |  0.392885   |  0.64446    |  0.171583    |      0.822682   |  0.317545   |     0.120361  |     0.68027   |  0.449969  |
| KBLab/bert-base-swedish-cased-new            |  0.428989   |  0.753263   |  0.16292     |      0.75478    |  0.140347   |     0.0420433 |     0.457718  |  0.391437  |
| xlm-roberta-base                             |  0.364267   |  0.700577   |  0.185628    |      0.81276    |  0.182248   |    -0.177215  |     0.455751  |  0.360574  |
| SVM                                          |  0.285916   |  0.517739   |  0.000204149 |      0.239667   |  0.0422635  |     0.0549607 |    -0.0185516 |  0.160314  |
| Decision Tree                                |  0.139739   |  0.275261   |  0.0190077   |      0.184487   |  0.0398626  |    -0.0586309 |     0.0253913 |  0.0893024 |
| Random                                       |  0.00783217 |  0.00308632 | -0.0906326   |     -0.00659552 | -0.00954447 |     0.0165235 |    -0.13896   | -0.0311843 |
| Random Forest                                |  0.00537142 | -0.312481   | -0.411051    |      0.144037   |  0.00334587 |    -0.177215  |     0.0126684 | -0.105046  |
| MaxFreq/Avg                                  | -0.0517904  | -0.340028   | -0.534511    |     -0.00149617 | -0.332667   |    -0.177215  |    -0.30954   | -0.249607  |

### SweWinogender Results

| Model                                        |   SweWinogender (Parity) |   SweWinogender (Alpha) |
|----------------------------------------------|--------------------------|-------------------------|
| KBLab/megatron-bert-large-swedish-cased-165k |                 0.995192 |               -0.29472  |
| xlm-roberta-base                             |                 0.995192 |               -0.298024 |
| xlm-roberta-large                            |                 0.985577 |               -0.315893 |
| KBLab/megatron-bert-base-swedish-cased-600k  |                 0.990385 |               -0.320952 |
| KBLab/bert-base-swedish-cased-new            |                 1        |               -0.328369 |
| AI-Nordics/bert-large-swedish-cased          |                 1        |               -0.332265 |
| KB/bert-base-swedish-cased                   |                 1        |               -0.332265 |
| NbAiLab/nb-bert-base                         |                 0.990385 |               -0.332306 |

### Best Hyperparameters

The search space for all tasks and transformer models were the following:

```json
{
    "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5],
    "batch_size": [16, 32]
}
```

with the exception of `SweMNLI`, that had its search space reduced to the immense training set size:

```json
{
    "learning_rate": [1e-5, 4e-5],
    "batch_size": [16, 32]
}
```

Furthermore, all models use the following hyperparameters along with HuggingFace Trainer's default arguments:

```json
{
    "warmup_ratio": 0.06,
    "weight_decay": 0.1,  # 0.0 if gpt
    "num_train_epochs": 10,
    "fp16": true
}
```

`num_train_epochs=10` is the maximum epochs, using early stopping with `patience=5`.

#### Model Hyperparameters Tables

Below follow the selected Hyperparameters for each model and task, along with a standard deviation of the evaluation metric for the different configurations.

**AI-Nordics/bert-large-swedish-cased**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 2e-05 |   16 | 0.0120759  |
| DaLAJ         | 4e-05 |   32 | 0.0138314  |
| SweMNLI       | 1e-05 |   16 | 0.00832644 |
| SweParaphrase | 3e-05 |   16 | 0.00558327 |
| SweWiC        | 3e-05 |   32 | 0.0153254  |
| SweWinograd   | 3e-05 |   32 | 0.0308409  |
| Swedish FAQ   | 1e-05 |   32 | 0.0166277  |

**KB/bert-base-swedish-cased**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 2e-05 |   16 | 0.02115    |
| DaLAJ         | 2e-05 |   32 | 0.00690644 |
| SweMNLI       | 1e-05 |   32 | 0.0118903  |
| SweParaphrase | 4e-05 |   32 | 0.00267101 |
| SweWiC        | 2e-05 |   16 | 0.0111782  |
| SweWinograd   | 1e-05 |   16 | 0.0618928  |
| Swedish FAQ   | 1e-05 |   16 | 0.0258529  |

**KBLab/bert-base-swedish-cased-new**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 2e-05 |   16 | 0.010503   |
| DaLAJ         | 2e-05 |   16 | 0.00939234 |
| SweMNLI       | 1e-05 |   16 | 0.00648224 |
| SweParaphrase | 4e-05 |   16 | 0.0423114  |
| SweWiC        | 1e-05 |   32 | 0.171214   |
| SweWinograd   | 1e-05 |   32 | 0.132972   |
| Swedish FAQ   | 3e-05 |   32 | 0.144674   |

**KBLab/megatron-bert-base-swedish-cased-600k**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 4e-05 |   16 | 0.0215247  |
| DaLAJ         | 4e-05 |   16 | 0.0171051  |
| SweMNLI       | 1e-05 |   16 | 0.00194938 |
| SweParaphrase | 4e-05 |   16 | 0.00612823 |
| SweWiC        | 4e-05 |   16 | 0.0291987  |
| SweWinograd   | 4e-05 |   32 | 0.114922   |
| Swedish FAQ   | 3e-05 |   32 | 0.00878437 |

**KBLab/megatron-bert-large-swedish-cased-165k**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 3e-05 |   16 | 0.0126327  |
| DaLAJ         | 3e-05 |   32 | 0.0174812  |
| SweMNLI       | 1e-05 |   32 | 0.00384093 |
| SweParaphrase | 4e-05 |   16 | 0.00475201 |
| SweWiC        | 4e-05 |   32 | 0.0130878  |
| SweWinograd   | 3e-05 |   16 | 0.0664638  |
| Swedish FAQ   | 4e-05 |   16 | 0.00752451 |

**NbAiLab/nb-bert-base**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 4e-05 |   16 | 0.0263801  |
| DaLAJ         | 1e-05 |   16 | 0.00804185 |
| SweMNLI       | 1e-05 |   16 | 0.0108116  |
| SweParaphrase | 4e-05 |   32 | 0.00655906 |
| SweWiC        | 4e-05 |   32 | 0.0228019  |
| SweWinograd   | 4e-05 |   32 | 0.029244   |
| Swedish FAQ   | 3e-05 |   16 | 0.330018   |

**xlm-roberta-base**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 4e-05 |   16 | 0.0325399  |
| DaLAJ         | 2e-05 |   32 | 0.0173028  |
| SweMNLI       | 1e-05 |   16 | 0.0144903  |
| SweParaphrase | 1e-05 |   16 | 0.00433707 |
| SweWiC        | 1e-05 |   16 | 0.233132   |
| SweWinograd   | 2e-05 |   16 | 0          |
| Swedish FAQ   | 1e-05 |   16 | 0.352092   |

**xlm-roberta-large**

| Task          |    LR |   BS |    hps std |
|---------------|-------|------|------------|
| ABSA          | 2e-05 |   16 | 0.240555   |
| DaLAJ         | 1e-05 |   32 | 0.477851   |
| SweMNLI       | 1e-05 |   32 | 0.471841   |
| SweParaphrase | 1e-05 |   16 | 0.00389993 |
| SweWiC        | 1e-05 |   32 | 0.31005    |
| SweWinograd   | 2e-05 |   32 | 0.128864   |
| Swedish FAQ   | 2e-05 |   32 | 0.454154   |


### Average Standard Deviation of HPS

The following table shows the average standard deviation of the hyperparameter configuration performances. Sorted on `avg std` and could indicate hyperparameter sensitivity of the models.

| Model                                        |   avg std |
|----------------------------------------------|-----------|
| AI-Nordics/bert-large-swedish-cased          | 0.0146587 |
| KBLab/megatron-bert-large-swedish-cased-165k | 0.017969  |
| KB/bert-base-swedish-cased                   | 0.0202202 |
| KBLab/megatron-bert-base-swedish-cased-600k  | 0.0285161 |
| NbAiLab/nb-bert-base                         | 0.0619795 |
| KBLab/bert-base-swedish-cased-new            | 0.0739355 |
| xlm-roberta-base                             | 0.0934135 |
| xlm-roberta-large                            | 0.298174  |


### Average Mean Distance to Max Metric

The following table shows the average of the mean distances to the maximum achieved performance.
I.e. for each task hyperparameter search, take the mean of the metric distances to the maximum hyperparameter configuration.

| Model                                        |   avg mean distance |
|----------------------------------------------|---------------------|
| KBLab/megatron-bert-large-swedish-cased-165k |           0.0249956 |
| AI-Nordics/bert-large-swedish-cased          |           0.026842  |
| KB/bert-base-swedish-cased                   |           0.0322925 |
| KBLab/megatron-bert-base-swedish-cased-600k  |           0.0442349 |
| NbAiLab/nb-bert-base                         |           0.0521342 |
| KBLab/bert-base-swedish-cased-new            |           0.0709957 |
| xlm-roberta-base                             |           0.0952137 |
| xlm-roberta-large                            |           0.360486  |


## Notes / Deviations

### SweFAQ Non-transformer baselines
The random baseline takes random values from the range of all seen labels, not the current number of possible answers.

Traditional ML baselines take a random answer from the candidates that the models independently predict as a correct answer.

### SweMNLI Traditional ML baselines
For these traditional ML baselines, only 5% (20,000 samples) of the training set is used for training. 
This did not seem to have a noticable effect on the end performance, and the motivation for this was to reduce the training time.
