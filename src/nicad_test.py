import multiprocessing
import os
import pickle
import tokenize
from io import BytesIO
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


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
    try:
        script = open(script_path, "r").read()

        comments, docstrings = extract_comments_and_docstrings(script)

        comment_lines = len(comments)
        number_of_comment_chars = sum([len(comment) for comment in comments])

        docstring_lines = len(docstrings)
        number_of_docstring_chars = sum([len(docstring) for docstring in docstrings])

        code_lines = len(script.split("\n"))
        number_of_code_chars = (
            len(script) - number_of_comment_chars - number_of_docstring_chars
        )

        return (
            number_of_comment_chars + number_of_docstring_chars
        ) / number_of_code_chars
    except Exception as e:
        return 0


nicad_ds_sample = 10
nicad_ds = pd.read_csv(f"/store/travail/vamaj/TWMC/NiCAD_results_{nicad_ds_sample}.csv")

accuracy = 0
recall = 0
precision = 0
f1 = 0

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
skipped = 0

for row in nicad_ds.iterrows():
    repo_name = row[1]["filename"]
    nicad_predicted = 1 if row[1]["repo_detected"] else 0
    # get a list of all python files in the repo, repo may contain subdirectories
    files = []
    for root, subdirs, fs in os.walk(os.path.join(os.getcwd(), "data", repo_name)):
        for f in fs:
            if f.endswith(".py"):
                files.append(os.path.join(root, f))

    actual = (
        1
        if any(0.01 <= value <= 0.8 for value in list(map(comment_to_code_ratio, files)))
        else 0
    )
    if actual == nicad_predicted:
        if actual == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        if nicad_predicted == 1:
            false_positives += 1
        else:
            false_negatives += 1

accuracy = (true_positives + true_negatives) / (
    true_positives + true_negatives + false_positives + false_negatives
)
recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1: {f1}")
print(f"sensitivity: {true_positives / (true_positives + false_negatives)}")
print(f"specificity: {true_negatives / (true_negatives + false_positives)}")
