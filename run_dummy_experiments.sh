#!/bin/bash

export PYTHONPATH="${pwd}:$PYTHONPATH"
# pip install evaluate
run_cmd="python3 dummy_baseline/dummy_experiment_driver.py"

echo $run_cmd
$run_cmd

echo "Done"
