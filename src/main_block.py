import argparse
import json
import logging
import logging.config
import os
import random
import sys
import warnings

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from checker import Checker, CheckerBlock
from models import SantaCoder, SantaCoderBlock

# Disable all warnings
warnings.filterwarnings("ignore")
# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(
    description="Trained Without My Consent - Block Generator"
)
parser.add_argument(
    "--language",
    type=str,
    default="py",
    help="language of the code",
)  # programming language
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data",
    help="path to the dataset",
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
    default="0",
    help="run id for the experiment",
)
parser.add_argument(
    "--working_dir",
    type=str,
    default=os.getcwd(),
    help="working directory",
)
args = parser.parse_args()


WORKING_DIR = os.getcwd() if args.working_dir == os.getcwd() else args.working_dir
print("\033[93m" + f"Working directory: {WORKING_DIR}" + "\033[0m")

model = SantaCoderBlock()


def get_model_output_inspector(file_path: str, run_num: int):
    """
    Invoke the model to generate outputs for the given script

    Args:
        file_path (str): path to the file
        run_num (int): run number id
    """
    global model
    results = []
    file_checker = CheckerBlock(file_path)  # initialize the checker
    model_inputs = (
        file_checker.prepare_inputs_for_prediction()
    )  # prepare the inputs for the model

    for candidate_input in tqdm(model_inputs):
        try:
            model_output = model.predict(
                candidate_input["prefix"], candidate_input["suffix"]
            )
            candidate_input["model_output"] = model_output
            results.append(candidate_input)
        except (RuntimeError, IndexError) as e:
            #! Wehn a CUDA error occurs, the error cascades and recovery is not possible. Hence, we restart the pipeline
            print("Dreaded CUDA ERROR. No recovery possible. Pipeline restart initiated.")
            # keep track of the files that caused CUDA error so that we can skip them in the next run
            with open(
                os.path.join(
                    WORKING_DIR,
                    "run_results",
                    f"assert_errors.txt",
                ),
                "a",
            ) as f:
                f.write(file_path + "\n")

            sys.exit(2)
    # save the results
    with open(
        os.path.join(
            os.getcwd(),
            "run_results",
            f"BlocksRun{args.run_num}",
            f"results_block_{run_num}.jsonl",
        ),
        "a",
    ) as f:
        json_results = json.dumps(results)
        f.write(json_results + "\n")


if __name__ == "__main__":
    # print all gpu devices available
    print("Available devices: ", torch.cuda.device_count())
    if torch.cuda.is_available():
        logging.info(f"GPU is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU is not available. Running on CPU")

    # create the directory to store the results
    if not os.path.exists(
        os.path.join(WORKING_DIR, "run_results", f"BlocksRun{args.run_num}")
    ):
        os.makedirs(os.path.join(WORKING_DIR, "run_results", f"BlocksRun{args.run_num}"))

    # get all the files in the dataset
    dataset_files = []
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(WORKING_DIR, args.dataset_path)
    ):
        python_files = [file for file in filenames if file.endswith(".py")]
        if python_files:
            dataset_files.extend(
                [os.path.join(WORKING_DIR, dirpath, file) for file in python_files]
            )

    # whether to go through the files in a descending or ascending order
    if args.sorted:
        dataset_files.sort(reverse=True)
    else:
        dataset_files.sort()

    # get the files that have already been processed so that we can skip them
    already_processed = open(
        os.path.join(WORKING_DIR, "run_results", "generated.txt"), "r"
    ).readlines()  # read already processed files
    # get the files that caused CUDA errors so that we can skip them
    dangerous_files = open(
        os.path.join(WORKING_DIR, "run_results", f"assert_errors.txt"),
        "r",
    ).readlines()

    already_processed = [file.rstrip("\n") for file in already_processed]
    dangerous_files = [file.rstrip("\n") for file in dangerous_files]

    for file_path in dataset_files:
        if file_path not in already_processed and file_path not in dangerous_files:
            results = []
            print("\033[91m" + file_path + "\033[0m")
            result = get_model_output_inspector(file_path, args.run_num)

            with open(
                os.path.join(WORKING_DIR, "run_results", "generated.txt"), "a"
            ) as f:
                f.write(file_path + "\n")
