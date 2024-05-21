import json
import multiprocessing as mp
import os
import random
import re
import shutil
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

JPLAG_DIR = os.path.join(
    os.getcwd(),
    "systems",
)
WORKING_DIR = os.path.join(os.getcwd())


def copy_python_files(src: str, dest: str) -> None:
    """
    Copies all Python files from src to dest.

    Args:
        src (str): Source directory
        dest (str): Destination directory
    """

    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            if filename.endswith(".py"):
                source_item = os.path.join(dirpath, filename)
                target_item = os.path.join(dest, src.split("/")[-1] + "_" + filename)
                target_dir = os.path.dirname(target_item)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy(source_item, target_item)


def process_directory(
    directory: str, selected_directories: list, core_number: int
) -> None:
    """
    Processes the files in a directory by running the JPlag colne detector on them.

    1 - Copies the files from the directory and the randomly selected directories to the JPlag directory
    2 - Runs JPlag on the files
    3 - Saves the results in a json file
    4 - Removes the files from the JPlag directory

    Args:
        directory (str): the main directory to run clone detection on
        selected_directories (list): randomly selected directories to run clone detection against
    """

    # Skip processing if the results already exist
    if os.path.exists(
        os.path.join(
            WORKING_DIR,
            "jplag_results",
            "original",
            f"jplag_results_{directory}.json",
        )
    ):
        return

    # change the working directory to the JPlag directory
    os.chdir(WORKING_DIR)
    source = os.path.join(os.getcwd(), "blocks", directory)
    target = os.path.join(JPLAG_DIR, f"analysis_target_{core_number}")

    # get the full paths of the randomly selected directories
    random_directory_paths = [
        os.path.join(os.getcwd(), "blocks", directory)
        for directory in selected_directories
    ]
    # align the prints so that they are easier to read
    print(
        "\033[92m"
        + f"{core_number}".ljust(2)
        + " - "
        + "Moving from TWMC to JPlag -> ".ljust(30)
        + os.path.join(JPLAG_DIR, "systems", directory).ljust(50)
        + "\033[0m"
    )
    # make a directory named analysis_target which will contain all the files to be analyzed
    os.makedirs(
        os.path.join(JPLAG_DIR, "systems", f"analysis_target_{core_number}"),
        exist_ok=True,
    )
    for path in [source] + random_directory_paths[:3]:
        copy_python_files(path, target)

    # change the working directory to the JPlag directory and run JPlag
    print(
        "\033[93m"
        + f"{core_number}".ljust(2)
        + " - "
        + f"Running JPlag Block on {directory}".ljust(50)
        + "\033[0m"
    )
    os.chdir(JPLAG_DIR)
    # no need to display terminal output
    os.system(
        f"java -jar {os.path.join(WORKING_DIR,'jplag.jar')} {os.path.join(JPLAG_DIR,f'analysis_target_{core_number}')} -l python3 -M RUN -r {os.path.join(JPLAG_DIR, 'systems', f'analysis_target_{core_number}')} --csv-export --cluster-skip",
    )
    # JPlag produces results as an html file, so we read the html file and save it as a json file. The reason for storing them and not processing them outright is to have a record of the results in case something goes wrong.
    csv_files = [
        file
        for file in os.listdir(
            os.path.join(JPLAG_DIR, "systems", f"analysis_target_{core_number}")
        )
        if file.endswith(".csv")
    ]
    JPlag_results = {}
    # read the original html file, process it and save it as a json file

    JPlag_results[f"analysis_target_{core_number}"] = pd.read_csv(
        os.path.join(
            JPLAG_DIR,
            "systems",
            f"analysis_target_{core_number}",
            "results.csv",
        )
    )
    trained_on_repos = pd.read_csv("/Users/ahvra/Nexus/TWMC/repo_trained_on.csv")
    for df in JPlag_results.values():
        positives = df[df["averageSimilarity"] > 0.3]
        negatives = df[df["averageSimilarity"] < 0.3]
        positives_originals = positives[
            positives["submissionName1"].str.contains("original")
        ]
        positives_generated = positives[
            positives["submissionName1"].str.contains("geenrated")
        ]
        negatives_originals = negatives[
            negatives["submissionName1"].str.contains("original")
        ]
        negatives_generated = negatives[
            negatives["submissionName1"].str.contains("generated")
        ]

        repos_train_on = pd.read_csv("/Users/ahvra/Nexus/TWMC/repo_trained_on.csv")

        true_positives = positives_originals.apply(
            lambda row: (
                repos_train_on.loc[
                    repos_train_on["repo_name"] == row["submissionName1"].split("_")[0],
                    "trained_on",
                ].values[0]
                if row["submissionName1"].split("_")[0]
                in repos_train_on["repo_name"].values
                else 0
            ),
            axis=1,
        )

        true_negatives = negatives_originals.apply(
            lambda row: (
                repos_train_on.loc[
                    repos_train_on["repo_name"] == row["submissionName1"].split("_")[0],
                    "trained_on",
                ].values[0]
                if row["submissionName1"].split("_")[0]
                in repos_train_on["repo_name"].values
                else 0
            ),
            axis=1,
        )

        false_positives = positives_originals.apply(
            lambda row: (
                1
                if row["submissionName1"].split("_")[0]
                in repos_train_on["repo_name"].values
                and repos_train_on.loc[
                    repos_train_on["repo_name"] == row["submissionName1"].split("_")[0],
                    "trained_on",
                ].values[0]
                == 0
                else 0
            ),
            axis=1,
        )

        false_negatives = negatives_originals.apply(
            lambda row: (
                1
                if row["submissionName1"].split("_")[0]
                in repos_train_on["repo_name"].values
                and repos_train_on.loc[
                    repos_train_on["repo_name"] == row["submissionName1"].split("_")[0],
                    "trained_on",
                ].values[0]
                == 1
                else 0
            ),
            axis=1,
        )
        # Calculate metrics
        precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
        recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (true_positives.sum() + true_negatives.sum()) / (true_positives.sum() + false_positives.sum() + false_negatives.sum() + true_negatives.sum())
        sensitivity = recall  # Sensitivity is the same as recall
        specificity = true_negatives.sum() / (true_negatives.sum() + false_positives.sum())

        # Print metrics
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"Accuracy: {accuracy}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")

    os.system(f"rm -rf {os.path.join(
            JPLAG_DIR,
            "systems",
            f"analysis_target_{core_number}",
            "results.csv",
        )}")
    


def worker_function(args):
    # number of randomly selected directories to run clone detection against
    NUM_SAMPLES = int(0.1 * len(os.listdir(os.path.join(os.getcwd(), "blocks"))))
    print(f"Number of samples: {NUM_SAMPLES}")

    directories_chunk, cpu_core_number = args

    directories = os.listdir(os.path.join(os.getcwd(), "blocks"))

    for directory in directories_chunk:
        selected_directories = random.sample(directories, NUM_SAMPLES)

        process_directory(directory, selected_directories, cpu_core_number)


if __name__ == "__main__":
    num_cores = mp.cpu_count()
    directories = os.listdir(os.path.join(os.getcwd(), "blocks"))

    # divide the directories into chunks and assign each chunk to a core
    chunk_size = len(directories) // num_cores
    directories_chunks = [
        directories[i : i + chunk_size] for i in range(0, len(directories), chunk_size)
    ]
    directories_with_core_numbers = [
        (chunk, i) for i, chunk in enumerate(directories_chunks)
    ]
    # process_directory(directories[0], directories[1:3], 0)
    # with mp.Pool(num_cores) as pool:
    #     #! Number of repo samples is set inside the worker function
    #     pool.map(worker_function, directories_with_core_numbers)
    #     pool.close()
    #     pool.join()
    process_directory(
        "/Users/ahvra/Nexus/TWMC/blocks/zhuhanqing",
        directories_with_core_numbers[0][0],
        directories_with_core_numbers[0][1],
    )
    # Directory containing the original JSON files
    original_directory_path = os.path.join(os.getcwd(), "jplag_results", "original")
    # Directory to save the result JSON files
    results_directory_path = os.path.join(os.getcwd(), "JPlag_results", "results")

