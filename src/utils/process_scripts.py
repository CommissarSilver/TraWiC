import json
import keyword
import logging
import multiprocessing as mp
import os

import tqdm

logger = logging.getLogger("process_scripts")


def get_word_count(script_path: str):
    """
    return a dictionary of the vocabulary frequency in the script

    Args:
        script_path (str): path to the script

    Returns:
        (dict): dictionary of the vocabulary frequency in the script
    """

    try:
        with open(script_path, "r") as f:
            script = f.read()
            script = script.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    except Exception as e:
        logger.exception(f"Error in opening the script at {script_path}")
        raise e

    try:
        splited_script = script.split()
        words = {word: splited_script.count(word) for word in splited_script}

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

    # get all the paths to the scripts
    scripts = [
        os.path.join(directory_path, i)
        for i in os.listdir(directory_path)
        if i.endswith(script_suffix)
    ]

    word_count = mp.Manager().dict()
    # get the word_count for each script using multiprocessing
    scripts_split = [scripts[i : i + 5000] for i in range(0, len(scripts), 5000)]
    with mp.Pool(mp.cpu_count()) as pool:
        for script_piece_num, script_piece in enumerate(
            tqdm.tqdm(scripts_split, total=len(scripts_split))
        ):
            results = pool.imap_unordered(get_word_count, script_piece)

            for result in results:
                for key, value in result.items():
                    word_count[key] = word_count.get(key, 0) + value

            # Save the word_count in JSON format
            json.dump(
                remove_keywords(dict(word_count)),
                open(
                    os.path.join(os.getcwd(), f"word_count_{script_piece_num}.json"), "w"
                ),
                indent=4,
            )

    # sort in descending order
    try:
        logger.info(
            f"Succesfully saved the word count in json format at {os.path.join(directory_path, 'word_count.json')}"
        )
    except Exception as e:
        logger.exception(
            f"Error in saving the word count in json format at {os.path.join(directory_path, 'word_count.json')}"
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
        logger.info(f"Succesfully removed Python builtins, variables and operators")
    except Exception as e:
        logger.exception(f"Error in saving the word count in json format")
        raise e

    return word_count


if __name__ == "__main__":
    import logging.config

    import yaml

    with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)

    mp.freeze_support()
    l = word_count_directory(
        directory_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
        script_suffix=".py",
    )
