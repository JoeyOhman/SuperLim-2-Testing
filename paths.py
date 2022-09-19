import os
import pathlib

ABSOLUTE_PATH_PROJ = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(ABSOLUTE_PATH_PROJ, "data")
TSV_PATH = os.path.join(DATASET_PATH, "tsv")
JSONL_PATH = os.path.join(DATASET_PATH, "jsonl")
