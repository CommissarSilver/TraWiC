import json
import os
import random
import tokenize
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import tqdm
from fuzzywuzzy import fuzz

try:
    from skip_data import SKIPS
except ImportError:
    from src.data.skip_data import SKIPS

sensitivity = False
sensitivity_threshold = 0.9
syn_thresh = 100
sem_thresh = 70


def build_dataset(jsonl_file_path: str) -> str:
    """
    Builds a dataset from a given jsonl file.
    Here we aim on cosolidating all of the data from the jsonl runs.
    We also add a column to the dataset that indicates whether the file was in the training set or not based on the comment to code ratio.

    Args:
        path_to_jsonl (str): path to the jsonl file to build the dataset from.
    """
    dataset = pd.DataFrame(
        columns=[
            "file_name",
            "level",
            "similarity_metric",
            "result",
            "similarity_objective",
            "model_output",
            "trained_on",
        ]
    )

    jsonl_file = open(os.path.join(jsonl_file_path, "results_all.jsonl"), "r")
    results_data = [line for line in jsonl_file]

    series_list = []
    for i, row in tqdm.tqdm(enumerate(results_data), total=len(results_data)):
        file_contents = json.loads(row)
        for entry in file_contents:
            try:
                file_name = entry["file_path"].split("data", 1)[1][1:]
                level = entry["level"]
                similarity_metric = entry["similarity_metric"]
                result = entry["result"]
                similarity_objective = entry["similarity_objective"]
                if similarity_objective in SKIPS:
                    raise KeyError
                model_output = entry["model_output"]
                # file_in_training_set = file_labels[file_name]

                series_list.append(
                    pd.Series(
                        {
                            "file_name": file_name,
                            "level": level,
                            "similarity_metric": similarity_metric,
                            "result": result,
                            "similarity_objective": similarity_objective,
                            "model_output": model_output,
                            "trained_on": (
                                0 if "not_trained_on" in jsonl_file_path else 1
                            ),
                        }
                    )
                )
            except KeyError:
                # there are some files that we can't calculate ratio for because they have some problems when read by tokenize. we skip them. they're not that many.
                continue
    dataset = pd.concat(series_list, axis=1).T
    # remove duplicates
    dataset = dataset.drop_duplicates()
    dataset.to_csv(
        os.path.join(jsonl_file_path, "dataset.csv"),
        index=False,
    )
    return os.path.join(jsonl_file_path, "dataset.csv")


def check_similarity(
    row: pd.Series,
    similairty_threshold: int = 60,
) -> int:
    """
    Checks the syntax similarity of a given model output with the similarity objective as ground truth.

    Args:
        row (pd.Series): row of the dataset to check similarity for.
        similairty_threshold (int, optional): threshold for considering similarity a success or not. Defaults to 60.
    """
    similarity_objective = (
        row["similarity_objective"].strip("\n").strip("\t").strip(" ")
        if not pd.isna(row["similarity_objective"])
        else ""
    )
    model_output = (
        row["model_output"].strip("\n").strip("\t").strip(" ")
        if not pd.isna(row["model_output"])
        else ""
    )
    similarity = fuzz.ratio(similarity_objective, model_output)

    return 1 if similarity >= similairty_threshold else 0


def check_similarity_sensitive(
    row: pd.Series,
    similairty_threshold: int = 60,
) -> int:
    """
    Simulate noise in the model output by checking the similarity of a given model output with the similarity objective as ground truth.

    args:
        row (pd.Series): row of the dataset to check similarity for.
        similairty_threshold (int, optional): threshold for considering similarity a success or not. Defaults to 60.
    """
    similarity_objective = (
        row["similarity_objective"].strip("\n").strip("\t").strip(" ")
        if not pd.isna(row["similarity_objective"])
        else ""
    )
    model_output = (
        row["model_output"].strip("\n").strip("\t").strip(" ")
        if not pd.isna(row["model_output"])
        else ""
    )
    similarity = fuzz.ratio(similarity_objective, model_output)

    #! This simulates noise only for the combined scenario
    if similarity >= similairty_threshold:
        if random.random() > sensitivity_threshold:
            return 1
        else:
            return 0
    else:
        return 0


def process_dataset(
    path_to_ds: str,
    syntax_threshold: int = 100,
    semantic_threshold: int = 60,
) -> None:
    """
    processes the csv dataset from build_dataset function and creates a final csv dataset that is ready to be used for training.

    Args:
        path_to_ds (str): path to dataset from build_dataset function.
        syntax_threshold (float, optional): threshold for syntax similarity. Defaults to 1.0.
        semantic_threshold (int, optional): threshold for semantic similarity. Defaults to 60.
    """
    ds = pd.read_csv(path_to_ds)
    ds = ds.sample(frac=1).reset_index(drop=True)

    if not sensitivity:
        ds["result"] = ds.apply(
            lambda row: (
                check_similarity(row, syntax_threshold)
                if row["level"] in ["function_names", "class_names", "variable_names"]
                else check_similarity(row, semantic_threshold)
            ),
            axis=1,
        )
    else:
        ds["result"] = ds.apply(
            lambda row: (
                check_similarity_sensitive(row, syntax_threshold)
                if row["level"] in ["function_names", "class_names", "variable_names"]
                else check_similarity_sensitive(row, semantic_threshold)
            ),
            axis=1,
        )

    # group the dataset by file_name, level, similarity_metric
    grouped_ds = ds.groupby(
        ["file_name", "level", "similarity_objective", "model_output"]
    )

    lm_dict = {
        file_name: {
            "class_hits": 0,
            "class_nums_total": 0,
            "function_hits": 0,
            "function_nums_total": 0,
            "variable_hits": 0,
            "variable_nums_total": 0,
            "string_hits": 0,
            "string_nums_total": 0,
            "comment_hits": 0,
            "comment_nums_total": 0,
            "docstring_hits": 0,
            "docstring_nums_total": 0,
            "trained_on": 0,
        }
        for file_name in ds["file_name"].unique()
    }
    # iterate over groups, and for each group, extract the number of hits for each type of token

    for name, group in tqdm.tqdm(grouped_ds):
        # if similarity_metric is function_name, class_name, variable_name, if there is even one 1 in the results column, then count it as a hit
        if name[1] == "class_names":
            lm_dict[name[0]]["class_hits"] += group["result"].values[0]
            lm_dict[name[0]]["class_nums_total"] += 1
        elif name[1] == "function_names":
            lm_dict[name[0]]["function_hits"] += group["result"].values[0]
            lm_dict[name[0]]["function_nums_total"] += 1
        elif name[1] == "variable_names":
            lm_dict[name[0]]["variable_hits"] += group["result"].values[0]
            lm_dict[name[0]]["variable_nums_total"] += 1
        # if similarity_metric is string, comment, docstring, if there is even one entry with an L-distance score more than 60, then count it as a hit
        elif name[1] == "strings":
            lm_dict[name[0]]["string_hits"] += group["result"].values[0]
            lm_dict[name[0]]["string_nums_total"] += 1
        elif name[1] == "comments":
            lm_dict[name[0]]["comment_hits"] += group["result"].values[0]
            lm_dict[name[0]]["comment_nums_total"] += 1
        elif name[1] == "docstrings":
            lm_dict[name[0]]["docstring_hits"] += group["result"].values[0]
            lm_dict[name[0]]["docstring_nums_total"] += 1
        lm_dict[name[0]]["trained_on"] = 1 if 1 in group["trained_on"].values else 0
    # normalize the number of hits by the total number of tokens
    for file_name, file_dict in lm_dict.items():
        lm_dict[file_name]["class_hits"] = (
            lm_dict[file_name]["class_hits"] / lm_dict[file_name]["class_nums_total"]
            if lm_dict[file_name]["class_nums_total"] != 0
            else 0
        )
        lm_dict[file_name]["function_hits"] = (
            lm_dict[file_name]["function_hits"]
            / lm_dict[file_name]["function_nums_total"]
            if lm_dict[file_name]["function_nums_total"] != 0
            else 0
        )
        lm_dict[file_name]["variable_hits"] = (
            lm_dict[file_name]["variable_hits"]
            / lm_dict[file_name]["variable_nums_total"]
            if lm_dict[file_name]["variable_nums_total"] != 0
            else 0
        )
        lm_dict[file_name]["string_hits"] = (
            lm_dict[file_name]["string_hits"] / lm_dict[file_name]["string_nums_total"]
            if lm_dict[file_name]["string_nums_total"] != 0
            else 0
        )
        lm_dict[file_name]["comment_hits"] = (
            lm_dict[file_name]["comment_hits"]
            / lm_dict[file_name]["comment_nums_total"]
            if lm_dict[file_name]["comment_nums_total"] != 0
            else 0
        )
        lm_dict[file_name]["docstring_hits"] = (
            lm_dict[file_name]["docstring_hits"]
            / lm_dict[file_name]["docstring_nums_total"]
            if lm_dict[file_name]["docstring_nums_total"] != 0
            else 0
        )
    # convert lm_dict to a dataframe
    lm_ds = pd.DataFrame.from_dict(lm_dict, orient="index")

    return lm_ds


if __name__ == "__main__":
    model_name = "llama"
    paths = [
        f"/home/vamaj/scratch/TraWiC/run_results/{model_name}_epoch_1",
        f"/home/vamaj/scratch/TraWiC/run_results/{model_name}_epoch_2",
        f"/home/vamaj/scratch/TraWiC/run_results/{model_name}_epoch_3",
    ]
    for path_num, path in enumerate(paths):
        # build_dataset(path)
        processed_datasets = process_dataset(
            os.path.join(path, "dataset.csv"),
            syntax_threshold=syn_thresh,
            semantic_threshold=sem_thresh,
        )
        print("Datasets built.")

        final_dataset = processed_datasets
        train_df = final_dataset.iloc[: int(0.8 * len(final_dataset))]
        test_df = final_dataset.iloc[int(0.8 * len(final_dataset)) :]
        print("Processing datasets...")

        if not sensitivity:
            if not os.path.exists(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}",
                )
            ):
                os.mkdir(
                    os.path.join(
                        "/home/vamaj/scratch/TraWiC/",
                        "rf_data",
                        f"{model_name}",
                        f"syn{syn_thresh}_sem{sem_thresh}",
                    )
                )

            train_df.to_csv(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}",
                    f"train_{path_num+1}.csv",
                ),
            )
            test_df.to_csv(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}",
                    f"test_{path_num+1}.csv",
                ),
            )

        else:
            if not os.path.exists(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}_sen{sensitivity_threshold}",
                )
            ):
                os.mkdir(
                    os.path.join(
                        "/home/vamaj/scratch/TraWiC/",
                        "rf_data",
                        f"{model_name}",
                        f"syn{syn_thresh}_sem{sem_thresh}_sen{sensitivity_threshold}",
                    )
                )

            train_df.to_csv(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}_sen{sensitivity_threshold}",
                    f"train_{path_num+1}.csv",
                ),
            )
            test_df.to_csv(
                os.path.join(
                    "/home/vamaj/scratch/TraWiC/",
                    "rf_data",
                    f"{model_name}",
                    f"syn{syn_thresh}_sem{sem_thresh}",
                    f"test_{path_num+1}.csv",
                ),
            )
        print("Datasets processed.")
