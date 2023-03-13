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

Note: Winogender will be evaluated automatically when training/evaluating with SweMNLI.


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
| Model                                        |       ABSA |   Argumentation |      DaLAJ |     SweMNLI |   SweParaphrase |     SweWiC |   SweWinograd |   Swedish FAQ |      Avg ↑ |
|----------------------------------------------|------------|-----------------|------------|-------------|-----------------|------------|---------------|---------------|------------|
| KBLab/megatron-bert-large-swedish-cased-165k |  0.571564  |      0.946352   |  0.727593  |  0.776482   |       0.908897  |  0.312644  |     0.258708  |    0.883748   |  0.673248  |
| AI-Nordics/bert-large-swedish-cased          |  0.533768  |      0.936037   |  0.716007  |  0.746819   |       0.895458  |  0.289614  |     0.240471  |    0.83509    |  0.649158  |
| KB/bert-base-swedish-cased                   |  0.513309  |      0.929816   |  0.715983  |  0.720911   |       0.900527  |  0.408507  |     0.188203  |    0.747808   |  0.640633  |
| KBLab/megatron-bert-base-swedish-cased-600k  |  0.497364  |      0.936032   |  0.690426  |  0.744255   |       0.895231  |  0.299891  |     0.179137  |    0.835171   |  0.634688  |
| xlm-roberta-large                            |  0.554599  |      0.933846   |  0.71126   |  0.783054   |       0.908649  |  0.345008  |     0.151711  |    0.679484   |  0.633452  |
| NbAiLab/nb-bert-base                         |  0.440776  |      0.923843   |  0.638692  |  0.717407   |       0.87952   |  0.351808  |     0.177085  |    0.718147   |  0.60591   |
| KBLab/bert-base-swedish-cased-new            |  0.476216  |      0.915303   |  0.726241  |  0.727923   |       0.812494  |  0.225232  |     0.0651066 |    0.553061   |  0.562697  |
| xlm-roberta-base                             |  0.398698  |      0.917526   |  0.671511  |  0.720407   |       0.871351  |  0.253994  |    -0.251163  |    0.619826   |  0.525269  |
| SVM                                          |  0.336347  |      0.906789   |  0.501149  |  0.126261   |       0.175726  |  0.012941  |     0.0981801 |    0.0473189  |  0.275589  |
| Decision Tree                                |  0.206274  |      0.862748   |  0.298952  |  0.14797    |       0.223647  |  0.0405762 |    -0.0220252 |   -0.02735    |  0.216349  |
| Random                                       | -0.0566656 |      0.00489102 | -0.0357937 |  0.00267875 |      -0.033005  |  0.0568082 |     0.0924728 |   -0.118317   | -0.0108663 |
| Random Forest                                |  0.0120151 |     -0.256239   | -0.31043   | -0.255086   |       0.159126  |  0.0272561 |    -0.251163  |    0.00746468 | -0.108382  |
| MaxFreq/Avg                                  | -0.0309908 |     -0.256239   | -0.347135  | -0.343683   |      -0.0246646 | -0.332     |    -0.251163  |   -0.316185   | -0.237757  |

### Test Results
| Model                                        |        ABSA |   Argumentation |       DaLAJ |      SweMNLI |   SweParaphrase |      SweWiC |   SweWinograd |   Swedish FAQ |      Avg ↑ |
|----------------------------------------------|-------------|-----------------|-------------|--------------|-----------------|-------------|---------------|---------------|------------|
| KBLab/megatron-bert-large-swedish-cased-165k |  0.508299   |       0.627597  |  0.753261   |  0.231612    |      0.873878   |  0.307947   |     0.188953  |     0.777436  |  0.533623  |
| AI-Nordics/bert-large-swedish-cased          |  0.480036   |       0.563173  |  0.745449   |  0.240594    |      0.862311   |  0.316317   |     0.191522  |     0.718673  |  0.51476   |
| KB/bert-base-swedish-cased                   |  0.529183   |       0.555028  |  0.739715   |  0.179116    |      0.844865   |  0.37619    |     0.139458  |     0.640648  |  0.500525  |
| xlm-roberta-large                            |  0.51631    |       0.583698  |  0.737508   |  0.20472     |      0.881687   |  0.3672     |     0.0806007 |     0.583791  |  0.494439  |
| KBLab/megatron-bert-base-swedish-cased-600k  |  0.449322   |       0.562494  |  0.718029   |  0.217683    |      0.866812   |  0.277146   |     0.0614488 |     0.709154  |  0.482761  |
| NbAiLab/nb-bert-base                         |  0.389723   |       0.540602  |  0.64446    |  0.171583    |      0.822616   |  0.325909   |     0.120361  |     0.659844  |  0.459387  |
| KBLab/bert-base-swedish-cased-new            |  0.427938   |       0.553602  |  0.753263   |  0.16292     |      0.754713   |  0.140347   |     0.0420433 |     0.446627  |  0.410182  |
| xlm-roberta-base                             |  0.365947   |       0.497157  |  0.700577   |  0.185628    |      0.812797   |  0.181145   |    -0.177215  |     0.473112  |  0.379893  |
| SVM                                          |  0.285916   |       0.353759  |  0.517739   |  0.000204149 |      0.23909    |  0.0422635  |     0.0549607 |     0.0381895 |  0.191515  |
| Decision Tree                                |  0.117238   |       0.155629  |  0.268636   |  0.0132697   |      0.199644   |  0.0398626  |    -0.24      |     0.0399946 |  0.0742843 |
| Random                                       |  0.00783217 |       0.0132383 |  0.00702486 | -0.0906326   |     -0.0427819  | -0.00954447 |     0.0806007 |    -0.150356  | -0.0230774 |
| Random Forest                                |  0.00537142 |      -0.272389  | -0.312481   | -0.411051    |      0.142812   |  0.00334587 |    -0.177215  |     0.0318551 | -0.123719  |
| MaxFreq/Avg                                  | -0.0517904  |      -0.272389  | -0.340028   | -0.433837    |     -0.00149459 | -0.332667   |    -0.177215  |    -0.309699  | -0.23989   |

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
| Argumentation | 1e-05 |   32 | 0.00375441 |
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
| Argumentation | 3e-05 |   32 | 0.00774597 |
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
| Argumentation | 4e-05 |   32 | 0.463319   |
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
| Argumentation | 3e-05 |   16 | 0.0777753  |
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
| Argumentation | 4e-05 |   16 | 0.0226433  |
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
| Argumentation | 3e-05 |   16 | 0.0194445  |
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
| Argumentation | 2e-05 |   16 | 0.029516   |
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
| Argumentation | 3e-05 |   32 | 0.512098   |
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
| AI-Nordics/bert-large-swedish-cased          | 0.0132957 |
| KBLab/megatron-bert-large-swedish-cased-165k | 0.0185533 |
| KB/bert-base-swedish-cased                   | 0.018661  |
| KBLab/megatron-bert-base-swedish-cased-600k  | 0.0346735 |
| NbAiLab/nb-bert-base                         | 0.0566626 |
| xlm-roberta-base                             | 0.0854263 |
| KBLab/bert-base-swedish-cased-new            | 0.122608  |
| xlm-roberta-large                            | 0.324914  |


### Average Mean Distance to Max Metric

The following table shows the average of the mean distances to the maximum achieved performance.
I.e. for each task hyperparameter search, take the mean of the metric distances to the maximum hyperparameter configuration.

| Model                                        |   avg mean distance |
|----------------------------------------------|---------------------|
| KBLab/megatron-bert-large-swedish-cased-165k |           0.0242377 |
| AI-Nordics/bert-large-swedish-cased          |           0.0243757 |
| KB/bert-base-swedish-cased                   |           0.0294655 |
| KBLab/megatron-bert-base-swedish-cased-600k  |           0.0472022 |
| NbAiLab/nb-bert-base                         |           0.0486684 |
| xlm-roberta-base                             |           0.0871857 |
| KBLab/bert-base-swedish-cased-new            |           0.122801  |
| xlm-roberta-large                            |           0.35873   |


## Notes / Deviations

### SweFAQ Non-transformer baselines
The random baseline takes random values from the range of all seen labels, not the current number of possible answers.

Traditional ML baselines take a random answer from the candidates that the models independently predict as a correct answer.

### SweMNLI Traditional ML baselines
For these traditional ML baselines, only 5% (20,000 samples) of the training set is used for training. 
This did not seem to have a noticeable effect on the end performance, and the motivation for this was to reduce the training time.
