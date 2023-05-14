import os, tqdm, json, inspect
import logger_utils
from datasets import load_dataset

logger = logger_utils.CustomLogger("dataset.log")


def get_thestack_dataset(
    language: str = "python",
    save_directory: str = os.path.join(os.getcwd(), "data"),
    scripts_num: int = 10**4,
):
    """
    get the TheStack dataset.
    ! Requires huggingface'scli login

    Args:
        language (str, optional): which language to download. Defaults to "python".
        save_directory (str, optional): where to store the downloaded scripts. Defaults to os.path.join(os.getcwd(), "data").
        scripts_num (int, optional): number of scripts to download. Defaults to 10**4.
    """

    # for logging purposes. DO NOT CHANGE!
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)

    # we'll use streaming so that it doesn't go and download the entire thing
    try:
        dataset = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{language}",
            streaming=True,
            split="train",
        )
        logger.info(
            f"{frame_info.filename} - {frame_info.function} - Succesfully connected to huggingface's TheStack dataset"
        )
    except:
        logger.exception(
            f"{frame_info.filename} - {frame_info.function} - Error connecting to huggingface's TheStack dataset"
        )

    # create the directory if it doesn't exist
    try:
        if not os.path.exists(os.path.join(save_directory, "the_stack", language)):
            os.makedirs(os.path.join(save_directory, "the_stack", language))
        data_dir = os.path.join(save_directory, "the_stack", language)
        logger.info(
            f"{frame_info.filename} - {frame_info.function} - Succesfully created the directory for saving the scripts"
        )
    except:
        logger.exception(
            f"{frame_info.filename} - {frame_info.function} - Error in creating directory for saving the scripts"
        )

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
            logger.info(
                f"{frame_info.filename} - {frame_info.function} - Succesfully downloaded and stored {str(scripts_num)} scripts"
            )
        except:
            logger.exception(
                f"{frame_info.filename} - {frame_info.function} - Error in dowloading/storing the scripts"
            )


if __name__ == "__main__":
    get_thestack_dataset(scripts_num=10)
