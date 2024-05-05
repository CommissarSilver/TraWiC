import argparse
import json
import logging
import logging.config
import os
import random

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from checker import Checker, MistralChecker
from models import MistralCoder

# load logging configuration
with open(
    os.path.join("/home/vamaj/scratch/TraWiC", "src", "logging_config.yaml"), "r"
) as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(description="Trained Without My Consent")
parser.add_argument(
    "--language",
    type=str,
    default="py",
    help="language of the code",
)  # programming language
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/repos",
    help="path to the dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size",
)  # batch size
parser.add_argument(
    "--model",
    type=str,
    default="mistral_coder",
    help="model to use",
)
parser.add_argument(
    "--sorted",
    type=bool,
    default=False,
    help="sort the dataset",
)
parser.add_argument(
    "--run_num",
    type=str,
    default="1",
    help="run number",
)
parser.add_argument(
    "--working_dir",
    type=str,
    default="/home/vamaj/scratch/TraWiC",
    help="working directory",
)

args = parser.parse_args()

WORKING_DIR = args.working_dir
print("\033[93m" + f"Working directory: {args.working_dir}" + "\033[0m")

model = MistralCoder() if args.model == "mistral_coder" else None


def get_model_output(file_path):
    results = []
    file_checker = MistralChecker(file_path)
    model_inputs = [
        file_checker.prepare_inputs_for_infill(level=i)
        for i in [
            "function_names",
            "variable_names",
            "class_names",
            "comments",
            "docstrings",
            "strings",
        ]
    ]
    model_inputs = [input for sublist in model_inputs for input in sublist]

    if model_inputs == []:
        return None
    for candidate_input in tqdm(model_inputs):
        model_output = model.infill(
            (
                candidate_input["infill"],
                candidate_input["prefix"],
                candidate_input["suffix"],
                candidate_input["level"],
            )
        )
        if model_output == "too_many_tokens":
            f = open(
                os.path.join(args.working_dir, "run_results", "too_many_tokens.txt"),
                "a",
            )
            f.write(file_path + "\n")
            return None
        else:
            try:
                result = file_checker.check_similarity(
                    model_output,
                    candidate_input,
                    similiarity_metric=(
                        "exact"
                        if candidate_input["level"]
                        in ["function_names", "variable_names", "class_names"]
                        else "fuzzy"
                    ),
                )
                results.append(
                    {
                        "file_path": file_path,
                        "level": candidate_input["level"],
                        "similarity_metric": (
                            "exact"
                            if candidate_input["level"]
                            in ["function_names", "variable_names", "class_names"]
                            else "fuzzy"
                        ),
                        "result": result,
                        "similarity_objective": candidate_input["infill"],
                        "model_output": model_output,
                    }
                )
            except Exception as e:
                logging.error(e)
                return None
    with open(
        os.path.join(
            args.working_dir, "run_results", f"TokensRun{args.run_num}", "results.jsonl"
        ),
        "a",
    ) as f:
        json_results = json.dumps(results)
        f.write(json_results)
        f.write("\n")


if __name__ == "__main__":
    print(args.sorted)
    print(type(args.run_num))
    print("Available devices: ", torch.cuda.device_count())

    if torch.cuda.is_available():
        logging.info(f"GPU is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU is not available. Running on CPU")

    if not os.path.exists(
        os.path.join(args.working_dir, "run_results", f"TokensRun{args.run_num}")
    ):
        os.mkdir(
            os.path.join(args.working_dir, "run_results", f"TokensRun{args.run_num}")
        )

    dataset_files = []
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(args.working_dir, args.dataset_path)
    ):
        python_files = [file for file in filenames if file.endswith(".py")]
        if python_files:
            dataset_files.extend(
                [os.path.join(args.working_dir, dirpath, file) for file in python_files]
            )

    dataset_files = (
        sorted(dataset_files) if args.sorted else sorted(dataset_files, reverse=True)
    )
    print(len(dataset_files))
    try:
        files_generated_blocks = open(
            os.path.join(args.working_dir, "run_results", "generated.txt"), "r"
        ).readlines()  # read already processed files

        files_generated_blocks = [file.rstrip("\n") for file in files_generated_blocks]

        files_generated_blocks = (
            sorted(files_generated_blocks)
            if args.sorted
            else sorted(files_generated_blocks, reverse=True)
        )
    except Exception as e:
        files_generated_blocks = []

    try:
        already_processed_files = open(
            os.path.join(args.working_dir, "run_results", "processed_tokens.txt"), "r"
        ).readlines()  # read already processed files
        already_processed_files = [
            file.rstrip("\n") for file in already_processed_files
        ]
    except Exception as e:
        already_processed_files = []

    try:
        dangerous_files = open(
            os.path.join(args.working_dir, "run_results", f"assert_errors.txt"),
            "r",
        ).readlines()
        dangerous_files = [file.rstrip("\n") for file in dangerous_files]
    except Exception as e:
        dangerous_files = []

    for file_path in dataset_files:
        if (
            file_path not in already_processed_files
            and file_path not in dangerous_files
        ):
            results = []
            print("\033[91m" + file_path + "\033[0m")
            result = get_model_output(file_path)

            with open(
                os.path.join(args.working_dir, "run_results", "processed_tokens.txt"),
                "a",
            ) as f:
                f.write(file_path + "\n")
