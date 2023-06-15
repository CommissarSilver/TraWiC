import tokenize, json, os
import pandas as pd
import tqdm
from io import BytesIO
from typing import List, Tuple


def extract_comments_and_docstrings(script: str) -> Tuple[List, List]:
    """
    Extracts comments and docstrings from a given script.

    Args:
        script (str): The script to extract comments and docstrings from.

    Returns:
        Tuple(List, List): A tuple containing a list of comments and a list of docstrings.
    """
    comments = []
    docstrings = []
    tokens = tokenize.tokenize(BytesIO(script.encode("utf-8")).readline)

    for token in tokens:
        if token.type == tokenize.COMMENT:
            comments.append(token.string.strip())
        elif token.type == tokenize.STRING and token.string.startswith(('"""', "'''")):
            docs = token.string.strip().split("\n")
            # remove the """ or ''' from the first and last lines
            docs = docs[1:-1]
            # append every element of docs to docstrings
            for element in docs:
                docstrings.append(element.strip())

    return comments, docstrings


def comment_to_code_ratio(script_path: str) -> float:
    """
    Calculates the ratio of comments and docstrings to code in a given script.

    Args:
        script_path (str): Path to the script to calculate the ratio for.

    Returns:
        float: ratio of comments and docstrings to code in the script.
    """
    script = open(script_path, "r").read()

    comments, docstrings = extract_comments_and_docstrings(script)

    comment_lines = len(comments)
    docstring_lines = len(docstrings)
    code_lines = len(script.split("\n"))

    return (comment_lines + docstring_lines) / code_lines


def build_dataset(path_to_jsonl: str) -> None:
    """
    Builds a dataset from a given jsonl file.
    Here we aim on cosolidating all of the data from the jsonl runs.
    We also add a column to the dataset that indicates whether the file was in the training set or not based on the comment to code ratio.

    Args:
        path_to_jsonl (str): path to the jsonl file to build the dataset from.
    """
    jsonl_ds = open(path_to_jsonl, "r").readlines()

    final_dataset = pd.DataFrame(
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

    trained_ons = pd.read_csv("/Users/ahura/Nexus/TWMC/src/trained_on.csv")
    file_labels = {
        file_name: 1
        if 0.01
        < comment_to_code_ratio(
            os.path.join(os.getcwd(), "data", "the_stack", "python", file_name)
        )
        < 0.8
        else 0
        for file_name in trained_ons["file_name"]
    }

    for i, row in tqdm.tqdm(enumerate(jsonl_ds), total=len(jsonl_ds)):
        file_contents = json.loads(row)
        for entry in file_contents:
            try:
                file_name = entry["file_path"].split("/")[-1]
                level = entry["level"]
                similarity_metric = entry["similarity_metric"]
                result = entry["result"]
                similarity_objective = entry["similarity_objective"]
                model_output = entry["model_output"]
                file_in_training_set = file_labels[file_name]

                final_dataset.loc[i] = [
                    file_name,
                    level,
                    similarity_metric,
                    result,
                    similarity_objective,
                    model_output,
                    file_in_training_set,
                ]
            except KeyError:
                # there are some files that we can't calculate ratio for because they have some problems when read by tokenize. we skip them. they're not that many.
                continue

    final_dataset.to_csv(
        os.path.join(os.getcwd(), "Runs", "final_dataset.csv"), index=False
    )

    print("Number of positives:", len([i for i in file_labels.values() if i == 1]))
    print("Number of negatives:", len([i for i in file_labels.values() if i == 0]))


def process_dataset(path_to_ds: str) -> None:
    """
    processes the csv dataset from build_dataset function and creates a final csv dataset that is ready to be used for training.

    Args:
        path_to_ds (str): path to dataset from build_dataset function.
    """
    ds = pd.read_csv(path_to_ds)
    # group the dataset by file_name, level, similarity_metric
    grouped_ds = ds.groupby(["file_name", "level", "similarity_objective"])

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
            lm_dict[name[0]]["class_hits"] += 1 if 1 in group["result"].values else 0
            lm_dict[name[0]]["class_nums_total"] += 1
        elif name[1] == "function_names":
            lm_dict[name[0]]["function_hits"] += 1 if 1 in group["result"].values else 0
            lm_dict[name[0]]["function_nums_total"] += 1
        elif name[1] == "variable_names":
            lm_dict[name[0]]["variable_hits"] += 1 if 1 in group["result"].values else 0
            lm_dict[name[0]]["variable_nums_total"] += 1
        # if similarity_metric is string, comment, docstring, if there is even one entry with an L-distance score more than 60, then count it as a hit
        elif name[1] == "strings":
            lm_dict[name[0]]["string_hits"] += (
                1 if any(i > 60 for i in group["result"].values) else 0
            )
            lm_dict[name[0]]["string_nums_total"] += 1
        elif name[1] == "comments":
            lm_dict[name[0]]["comment_hits"] += (
                1 if any(i > 60 for i in group["result"].values) else 0
            )
            lm_dict[name[0]]["comment_nums_total"] += 1
        elif name[1] == "docstrings":
            lm_dict[name[0]]["docstring_hits"] += (
                1 if any(i > 60 for i in group["result"].values) else 0
            )
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
            lm_dict[file_name]["comment_hits"] / lm_dict[file_name]["comment_nums_total"]
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
    # save the dataframe to a csv file
    lm_ds.to_csv("lm_dataset.csv")


if __name__ == "__main__":
    build_dataset("/Users/ahura/Nexus/TWMC/Runs/file.jsonl")
    process_dataset("/Users/ahura/Nexus/TWMC/Runs/Run 02/final_dataset.csv")
