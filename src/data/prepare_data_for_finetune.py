from checker import Checker
import json
import os
from tqdm import tqdm
import math


def get_python_files_list(dir_path):
    python_files = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        python_files += [
            os.path.join(dirpath, file) for file in filenames if file.endswith(".py")
        ]
    return python_files


all_files = get_python_files_list("/Users/ahvra/Nexus/TWMC/repos")
checkers = [Checker(file) for file in all_files]
all_candidates = []
for checker in tqdm(checkers, desc="Processing checkers", total=len(checkers)):
    candidates = [
        checker.prepare_inputs_for_infill(level=i)
        for i in tqdm(
            [
                "function_names",
                "variable_names",
                "class_names",
                "comments",
                "docstrings",
                "strings",
            ],
            desc="Processing levels",
            leave=False,
        )
    ]
    all_candidates += [
        candidate for sublist in candidates for candidate in sublist if candidate
    ]

num_records_per_json = 2000
num_jsons = math.ceil(len(all_candidates) / num_records_per_json)

for i in range(num_jsons):
    start_index = i * num_records_per_json
    end_index = min((i + 1) * num_records_per_json, len(all_candidates))
    json_data = all_candidates[start_index:end_index]

    json_filename = (
        f"/Users/ahvra/Nexus/TWMC/data/finetune_ds/finetune_candidates_{i+1}.json"
    )
    with open(json_filename, "w") as f:
        json.dump(json_data, f)
