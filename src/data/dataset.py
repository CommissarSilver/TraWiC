import json
import logging
import os
import tokenize
from io import BytesIO
from typing import List, Tuple

import tqdm
from datasets import load_dataset

logger = logging.getLogger("process_scripts")

EXCLUDED_DS = [
    "openai/human-eval",
    "hendrycks/apps",
    "google-research/google-research",
    "nuprl/MultiPL-E",
]  # these repos are excluded because they were excluded from SantaCoder's training dataset


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
        number_of_comment_chars= sum([len(comment) for comment in comments])
        
        docstring_lines = len(docstrings)
        number_of_docstring_chars = sum([len(docstring) for docstring in docstrings])
        
        code_lines = len(script.split("\n"))
        number_of_code_chars = len(script) - number_of_comment_chars - number_of_docstring_chars
        

        return (number_of_comment_chars + number_of_docstring_chars) / number_of_code_chars
    except Exception as e:
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
        logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except Exception as e:
        logger.exception(f"Error connecting to huggingface's TheStack dataset")
        raise e
    # create the directory if it doesn't exist
    try:
        if not os.path.exists(os.path.join(save_directory, "the_stack", language)):
            os.makedirs(os.path.join(save_directory, "the_stack", language))
        data_dir = os.path.join(save_directory, "the_stack", language)

        logger.info(f"Succesfully created the directory for saving the scripts")
    except Exception as e:
        logger.exception(f"Error in creating directory for saving the scripts")
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
            logger.info(f"Succesfully downloaded and stored {str(scripts_num)} scripts")
        except:
            logger.exception(f"Error in dowloading/storing the scripts")


def get_python_repos_info(dataset_list: dict) -> None:
    """
    Unlike the get_thestack_dataset function, this function gets entire repositories instead of just scripts.
    The purpose of this function is to only get repository names and whether they are in the original training set of not.
    IT DOES NOT DIRECTLZ DOWNLAOD THE REPOSITORIES

    Args:
        dataset_list (dict): a dictionary of repository names and whether they are in the original training set of not
    Raises:
        e: error in connecting to HuggingFace's TheStack dataset
    """
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
        logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except Exception as e:
        logger.exception(f"Error connecting to huggingface's TheStack dataset")
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
            logging.info("processed {} repos".format(len(repos.keys())))
            json.dump(repos, open(f"repos_info_{len(repos.keys())}.json", "w"))


def get_repos() -> None:
    """
    Downloads the repositories that have been marked as study candidates

    Raises:
        e: error in connecting to HuggingFace's TheStack dataset
    """
    x = open(os.path.join(os.getcwd(), "data", "repos_alot.json"), "r").read()
    repos_info = json.loads(x)

    try:
        dataset = load_dataset(
            "bigcode/the-stack",
            revision="v1.1",
            data_dir=f"data/python",
            streaming=True,
            split="train",
        )
        logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except Exception as e:
        logger.exception(f"Error connecting to huggingface's TheStack dataset")
        raise e

    for dataset_sample in iter(dataset):
        if dataset_sample["max_stars_repo_name"] in repos_info.keys():
            if (
                repos_info[dataset_sample["max_stars_repo_name"]]["in_train_num"]
                and repos_info[dataset_sample["max_stars_repo_name"]]["scripts_num"] >= 1
            ):  # check that the repository has more than one script and that it has at least one script in the original training set
                if not os.path.exists(
                    os.path.join(
                        os.getcwd(), "data", dataset_sample["max_stars_repo_name"]
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            os.getcwd(), "data", dataset_sample["max_stars_repo_name"]
                        )
                    )
                with open(
                    os.path.join(
                        os.getcwd(),
                        "data",
                        dataset_sample["max_stars_repo_name"],
                        dataset_sample["max_stars_repo_path"].split("/")[-1],
                    ),
                    "w",
                ) as f:  # save the script
                    f.write(dataset_sample["content"])


if __name__ == "__main__":
    import argparse
    import logging.config

    import pandas as pd
    import yaml

    with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    parser = argparse.ArgumentParser(
        description="Trained Without My Consent - Dataset Module"
    )
    parser.add_argument(
        "--get_scripts",
        type=bool,
        default=False,
        help="Whether to download individual scripts or entire repositories",
    )  # programming language
    args = parser.parse_args()

    if args.get_scripts:
        get_thestack_dataset(scripts_num=10**5)
    else:
        # repo_info = json.load(open(os.path.join(os.getcwd(), "data", "repos.json"), "r"))

        # get_python_repos_info(repo_info)
        get_repos()
