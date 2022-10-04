import csv
import os
from pathlib import Path
import json
from glob import glob

from paths import TSV_PATH, JSONL_PATH

if __name__ == '__main__':
    tsv_files = glob(os.path.join(TSV_PATH, "**/*.tsv"), recursive=True)
    # jsonl_files = [file_path.replace(TSV_PATH, JSONL_PATH) for file_path in tsv_files]

    for tsv_file in tsv_files:
        out_file = tsv_file.replace(TSV_PATH, JSONL_PATH).replace(".tsv", ".jsonl")

        # Parse input data
        with open(tsv_file, 'r') as f:
            # Had to remove " as quotechar for one particular sample to work, which included a single double-quote
            read_tsv = csv.reader(f, delimiter="\t", quotechar=None)
            # read_tsv = csv.reader(f, delimiter="\t")

            rows = []
            for row in read_tsv:
                rows.append(row)

        headers = rows[0]
        samples = rows[1:]

        # Create json objects
        json_objects = []
        for sample in samples:
            obj = {
                k: v for k, v in zip(headers, sample)
            }
            json_objects.append(obj)

        # Create output directory
        out_dir_name = os.path.dirname(out_file)
        Path(out_dir_name).mkdir(parents=True, exist_ok=True)

        # Write output file
        with open(out_file, 'w') as f:
            for obj in json_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")



