import os, tqdm, json, logging
from datasets import load_dataset
import tokenize, json, os
import pandas as pd
import tqdm
from io import BytesIO
from typing import List, Tuple

# from skip_daat import SKIPS


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


def comment_to_code_ratio(script: str) -> float:
    """
    Calculates the ratio of comments and docstrings to code in a given script.

    Args:
        script_path (str): Path to the script to calculate the ratio for.

    Returns:
        float: ratio of comments and docstrings to code in the script.
    """
    try:
        # script = open(script_path, "r").read()

        comments, docstrings = extract_comments_and_docstrings(script)

        comment_lines = len(comments)
        docstring_lines = len(docstrings)
        code_lines = len(script.split("\n"))

        return (comment_lines + docstring_lines) / code_lines
    except Exception:
        return 2


def get_thestack_dataset(
    language: str = "python",
    save_directory: str = os.path.join(os.getcwd(), "data"),
    scripts_num: int = 10**4,
) -> None:
    """
    get the TheStack dataset.
    ! Requires huggingface's cli login

    Args:
        language (str, optional): which language to download. Defaults to "python".
        save_directory (str, optional): where to store the downloaded scripts. Defaults to os.path.join(os.getcwd(), "data").
        scripts_num (int, optional): number of scripts to download. Defaults to 10**4.
    """

    # we'll use streaming so that it doesn't go and download the entire thing
    try:
        dataset = load_dataset(
            "bigcode/the-stack",
            revision="v1.1",
            data_dir=f"data/{language}",
            streaming=True,
            split="train",
        )
        # logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except Exception as e:
        # logger.exception(f"Error connecting to huggingface's TheStack dataset")
        raise e
    # create the directory if it doesn't exist
    try:
        if not os.path.exists(os.path.join(save_directory, "the_stack", language)):
            os.makedirs(os.path.join(save_directory, "the_stack", language))
        data_dir = os.path.join(save_directory, "the_stack", language)
        # logger.info(f"Succesfully created the directory for saving the scripts")
    except Exception as e:
        # logger.exception(f"Error in creating directory for saving the scripts")
        raise e

    i = 0
    # use tracker to index the hexshas of the stored scripts
    tracker = {}
    # use tqdm to visualize the progress bar
    with tqdm.tqdm(total=scripts_num) as pbar:
        try:
            for dataset_sample in iter(dataset):
                if (
                    dataset_sample["ext"] == "py"
                    and dataset_sample["max_stars_repo_name"] not in EXCLUDED_DS
                ):
                    with open(
                        os.path.join(
                            os.path.join(data_dir),
                            f"the_stack_{language}_script_{i}.{dataset_sample['ext']}",
                        ),
                        "w",
                    ) as f:
                        f.write(dataset_sample["content"])
                        tracker[
                            f"the_stack_{language}_script_{i}.{dataset_sample['ext']}"
                        ] = {
                            "number": str(i),
                            "hash": dataset_sample["hexsha"],
                            "stars_count": dataset_sample["max_stars_count"]
                            if dataset_sample["max_stars_count"] != None
                            else 0,
                        }

                    i += 1

                    pbar.update(1)

                if i == scripts_num:
                    json.dump(tracker, open(os.path.join(data_dir, "index.json"), "w"))
                    break
            # logger.info(f"Succesfully downloaded and stored {str(scripts_num)} scripts")
        except Exception as e:
            # logger.exception(f"Error in dowloading/storing the scripts")
            print(e)


def get_python_repos(dataset_list):
    repos = {}
    num_repos = 0

    try:
        dataset = load_dataset(
            "bigcode/the-stack",
            revision="v1.1",
            data_dir=f"data/python",
            streaming=True,
            split="train",
        )
        # logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except Exception as e:
        # logger.exception(f"Error connecting to huggingface's TheStack dataset")
        raise e

    for dataset_sample in iter(dataset):
        if (
            dataset_sample["ext"] == "py"
            and dataset_sample["max_stars_repo_name"] in dataset_list
        ):
            sample_comment_to_code_ratio = comment_to_code_ratio(
                dataset_sample["content"]
            )
            if dataset_sample["max_stars_repo_name"] not in repos.keys():
                repos[dataset_sample["max_stars_repo_name"]] = {
                    "scripts_num": 1,
                    "in_train_num": 1 if 0.01 < sample_comment_to_code_ratio < 0.8 else 0,
                }
            else:
                repos[dataset_sample["max_stars_repo_name"]]["scripts_num"] += 1
                repos[dataset_sample["max_stars_repo_name"]]["in_train_num"] += (
                    1 if 0.01 < sample_comment_to_code_ratio < 0.8 else 0
                )
        if len(repos.keys()) % 10000 == 0:
            print("processed {} repos".format(len(repos.keys())))
            json.dump(repos, open("repos_alot.json", "w"))

def get_repos_alot():
    x=open('/Users/ahura/Nexus/TWMC/data/repos_alot.json','r').read()
    repos_alot=json.loads(x)
    dataset = load_dataset(
            "bigcode/the-stack",
            revision="v1.1",
            data_dir=f"data/python",
            streaming=True,
            split="train",
        )
    for dataset_sample in iter(dataset):
        if dataset_sample["max_stars_repo_name"] in repos_alot.keys():
            if repos_alot[dataset_sample["max_stars_repo_name"]]["in_train_num"] and repos_alot[dataset_sample["max_stars_repo_name"]]["scripts_num"] >=1:
                if not os.path.exists(os.path.join('/Users/ahura/Nexus/TWMC/data',dataset_sample["max_stars_repo_name"])):
                    os.makedirs(os.path.join('/Users/ahura/Nexus/TWMC/data',dataset_sample["max_stars_repo_name"]))
                with open(
                        os.path.join(
                            '/Users/ahura/Nexus/TWMC/data',
                            dataset_sample["max_stars_repo_name"],
                            dataset_sample["max_stars_repo_path"].split('/')[-1]),
                        "w",) as f:
                            f.write(dataset_sample["content"])
    

if __name__ == "__main__":
    repo_info = pd.read_csv(
        "/Users/ahura/Nexus/TWMC/src/data/repo_info_more_than_10_less_than_50.csv"
    )
    # sample 10 percent of the repos
    # repo_info = repo_info.sample(frac=0.1)
    # get_python_repos(repo_info["repo_name"].tolist())
    get_repos_alot()
