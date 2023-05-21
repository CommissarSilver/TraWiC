import os, tqdm, json, logging
from datasets import load_dataset

logger = logging.getLogger("process_scripts")


def get_thestack_dataset(
    language: str = "python",
    save_directory: str = os.path.join(os.getcwd(), "data"),
    scripts_num: int = 10**4,
):
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
            data_dir=f"data/{language}",
            streaming=True,
            split="train",
        )
        logger.info(f"Succesfully connected to huggingface's TheStack dataset")
    except:
        logger.exception(f"Error connecting to huggingface's TheStack dataset")

    # create the directory if it doesn't exist
    try:
        if not os.path.exists(os.path.join(save_directory, "the_stack", language)):
            os.makedirs(os.path.join(save_directory, "the_stack", language))
        data_dir = os.path.join(save_directory, "the_stack", language)
        logger.info(f"Succesfully created the directory for saving the scripts")
    except:
        logger.exception(f"Error in creating directory for saving the scripts")

    i = 0
    # use tracker to index the hexshas of the stored scripts
    tracker = {}
    # use tqdm to visualize the progress bar
    with tqdm.tqdm(total=scripts_num) as pbar:
        try:
            for dataset_sample in iter(dataset):
                with open(
                    os.path.join(
                        os.path.join(data_dir),
                        f"the_stack_{language}_script_{i}.{dataset_sample['ext']}",
                    ),
                    "w",
                ) as f:
                    f.write(dataset_sample["content"])
                    tracker[dataset_sample["hexsha"]] = {
                        "number": str(i),
                        "name": f"the_stack_{language}_script_{i}.{dataset_sample['ext']}",
                    }

                i += 1

                pbar.update(1)

                if i == scripts_num:
                    json.dump(tracker, open(os.path.join(data_dir, "index.json"), "w"))
                    break
            logger.info(f"Succesfully downloaded and stored {str(scripts_num)} scripts")
        except:
            logger.exception(f"Error in dowloading/storing the scripts")


if __name__ == "__main__":
    import yaml, logging.config

    with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)

    get_thestack_dataset(scripts_num=10)
