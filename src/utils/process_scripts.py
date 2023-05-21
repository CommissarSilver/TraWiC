import os, tqdm, json, inspect, keyword, logger_utils as logger_utils
import multiprocessing as mp

logger = logger_utils.CustomLogger("process_scripts")


def get_word_count(script_path: str):
    """
    return a dictionary of the vocabulary frequency in the script

    Args:
        script_path (str): path to the script

    Returns:
        (dict): dictionary of the vocabulary frequency in the script
    """
    # for logging purposes. DO NOT CHANGE!

    try:
        with open(script_path, "r") as f:
            script = f.read()
            script = script.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    except Exception as e:
        logger.exception(f"Error in opening the script at {script_path}")
        raise e

    try:
        words = {}
        words = {
            word: 1 if not word in words.keys() else words[word] + 1
            for word in script.split(" ")
        }

        return words
    except Exception as e:
        logger.exception(
            f"Error in counting the script's word (vocabulary) frequency at {script_path}"
        )
        raise e


def word_count_directory(directory_path: str, script_suffix: str):
    """
    counts the entire directory's word (vocabulary) frequency

    Args:
        directory_path (str): path containing all the scripts
        scipt_suffix (str): suffix of the scripts. for example .py for python scripts

    Returns:
        (dict): word count of the entire directory
    """
    # for logging purposes. DO NOT CHANGE!
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)

    # get all the paths to the scripts
    scripts = [
        os.path.join(directory_path, i)
        for i in os.listdir(directory_path)
        if i.endswith(script_suffix)
    ]

    word_count = {}
    worker_pool = mp.Pool(mp.cpu_count() - 1)
    # get the word_count for each script using multiprocessing
    try:
        results = worker_pool.map_async(get_word_count, scripts).get()
    except Exception as e:
        logger.exception(
            f"{frame_info.filename} - {frame_info.function} - Error in counting the entire directory's word (vocabulary) frequency"
        )
        raise e
    # count the word_count of the entire directory
    for dictionary in results:
        for k in dictionary.keys():
            if k not in word_count.keys():
                word_count[k] = dictionary[k]
            else:
                word_count[k] += dictionary[k]
    # sort in descending order
    try:
        word_count = {
            k: v
            for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        }
        # save the word_count in json format
        json.dump(remove_keywords(word_count), open("word_count.json", "w"), indent=4)

        logger.info(
            f"{frame_info.filename} - {frame_info.function} - Succesfully saved the word count in json format"
        )
    except Exception as e:
        logger.exception(
            f"{frame_info.filename} - {frame_info.function} - Error in saving the word count in json format"
        )
        raise e

    return word_count


def remove_keywords(word_count: dict):
    """
    removes all the keywords and builtins from the word_count

    Args:
        word_count (dict): word_count of the entire directory

    Returns:
        (dict): word conut without keywords and builtins
    """
    # for logging purposes. DO NOT CHANGE!
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)

    builtins = set(
        dir(__builtins__)
        + keyword.kwlist
        + [
            "''",
            '""',
            "+",
            "=",
            "==",
            "+=",
            "-=",
            "*=",
            "/=",
            "<=",
            ">=",
            "!=",
            "-",
            "/",
            "%",
            "None",
            "|",
            "*",
            "**",
            "#",
            "<",
            ">",
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
            ":",
            "___",
            "{}",
            "[]",
            "()",
        ]
    )

    try:
        for builtin in builtins:
            try:
                del word_count[builtin]
            except KeyError:
                pass
        logger.info(
            f"{frame_info.filename} - {frame_info.function} - Succesfully removed Python builtins, variables and operators"
        )
    except Exception as e:
        logger.exception(
            f"{frame_info.filename} - {frame_info.function} - Error in saving the word count in json format"
        )
        raise e

    return word_count


if __name__ == "__main__":
    mp.freeze_support()
    l = word_count_directory(
        directory_path="/Users/ahura/Nexus/TWMC/data/the_stack/python/",
        script_suffix=".py",
    )
