import os, tqdm, json, logging
from datasets import load_dataset


logger = logging.getLogger("process_scripts")

EXCLUDED_DS = [
    "openai/human-eval",
    "hendrycks/apps",
    "google-research/google-research",
    "nuprl/MultiPL-E",
]


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


def get_python_repos():
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
            and dataset_sample["max_stars_repo_name"] not in EXCLUDED_DS
        ):
            if dataset_sample["max_stars_repo_name"] not in repos.keys():
                repos[dataset_sample["max_stars_repo_name"]] = 1
                num_repos += 1
            else:
                repos[dataset_sample["max_stars_repo_name"]] += 1

            if len(repos.keys()) % 10000 == 0:
                # save the dictionary to a json file
                json.dump(
                    repos,
                    open(
                        os.path.join(
                            os.getcwd(),
                            "data",
                            f"repos.json",
                        ),
                        "w",
                    ),
                )
                print(f"Saved {len(repos.keys())} repos")


if __name__ == "__main__":
    import yaml, logging.config

    with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)

    # get_thestack_dataset(scripts_num=10**5)
    get_python_repos()
