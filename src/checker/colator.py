import re, logging
from typing import Tuple, List, Dict

logger = logging.getLogger()


def prepare_input(input_string: str):
    """
    Extracts the following items from the input code script:
        - docstrings
        - comments
        - function names
        - class names
        - variable names
        - strings
    the results are stored in the `processed_input` attribute of the class

    """

    # use regex to extract the mentioned items
    docstrings_iter = re.finditer(r'"""[\s\S]*?"""', input_string)
    comments_iter = re.finditer(r"\s*#(.*)", input_string)
    function_names_iter = re.finditer(
        r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", input_string
    )
    class_names_iter = re.finditer(
        r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?", input_string
    )
    variable_names_iter = re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", input_string)
    strings_iter = re.finditer(r"\".*?\"|'.*?'", input_string)

    # for each of the items, we need to store their value and their line numbers
    try:
        docstrings = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): match.group()
            for match in docstrings_iter
        }
        logger.debug(f"Extracted docstrings: {docstrings}")
    except Exception as e:
        logger.exception(f"Error in extracting docstrings: {e}")
        docstrings = None

    try:
        comments = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): match.group()
            for match in comments_iter
        }
        logger.debug(f"Extracted comments: {comments}")
    except Exception as e:
        logger.exception(f"Error in extracting comments: {e}")
        comments = None

    try:
        function_names = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): (match.group(1), match.group(2))
            for match in function_names_iter
        }
        logger.debug(f"Extracted function_names: {function_names}")
    except Exception as e:
        logger.exception(f"Error in extracting function names:{e}")
        function_names = None

    try:
        class_names = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): (match.group(1), match.group(2))
            for match in class_names_iter
        }
        logger.debug(f"Extracted class_names: {class_names}")
    except Exception as e:
        logger.exception(f"Error in extracting class names: {e}")
        class_names = None

    try:
        variable_names = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): match.group(1)
            for match in variable_names_iter
        }
        logger.debug(f"Extracted variable_names: {variable_names}")
    except Exception as e:
        logger.exception(f"Error in extracting variable names: {e}")
        variable_names = None

    try:
        strings = {
            (
                input_string.count("\n", 0, match.start()) + 1,
                input_string.count("\n", 0, match.end()) + 1,
            ): match.group()
            for match in strings_iter
        }
        logger.debug(f"Extracted strings: {strings}")
    except Exception as e:
        logger.exception(f"Error in extracting strings: {e}")
        strings = None
    logger.debug("*" * 50)

    logger.info("Finished preparing input")

    return {
        "docstrings": docstrings,
        "comments": comments,
        "function_names": function_names,
        "class_names": class_names,
        "variable_names": variable_names,
        "strings": strings,
    }


def separate_script(script_text: str, word: str, line_number: int) -> Tuple[str, str]:
    """
    Separates the script into two parts, before and after the specified word in the specified line number

    Args:
        script_text (str): the text of the script
        word (str): the word to separate the script with
        line_number (int): the line number of the word

    Returns:
        (str, str): the prefix and suffix of the script
    """
    lines = script_text.split("\n")
    prefix = "\n".join(
        lines[: line_number - 1]
    )  # Prefix contains lines before the specified line number
    suffix = "\n".join(
        lines[line_number - 1 :]
    )  # Suffix contains lines from the specified line number onwards

    # Find the index of the word in the suffix and where it finishes
    word_index = suffix.find(word)
    word_end = word_index + len(word)

    prefix += suffix[:word_index]
    suffix = suffix[word_end:]

    return prefix, suffix


def prepare_inputs_for_infill(
    original_input_string: str,
    processed_input: dict,
) -> List[Dict]:
    """
    Prepares the input for the infill model

    Args:
        level (str): "fuinction_names", "class_names", "variable_names", "strings", "docstrings", "comments"

    Returns:
        List[Dict]: list of candidates for the infill model
    """
    candidates = {
        "function_names": [],
        "class_names": [],
        "variable_names": [],
        "strings": [],
        "docstrings": [],
        "comments": [],
    }

    for level in processed_input.keys():
        if processed_input[level] != None:
            model_input_candidates = processed_input[level]
        else:
            continue
        for key, item in model_input_candidates.items():
            if level in ("function_names", "class_names"):
                # for function and class names, we need to separate the script into prefix and suffix
                prefix, suffix = separate_script(
                    script_text=original_input_string, word=item[0], line_number=key[0]
                )
                candidates[level].append(
                    {
                        "infill": item[0],
                        "line": key[0],
                        "prefix": prefix,
                        "suffix": suffix,
                        "level": level,
                    }
                )
                # if the function has arguments, or the class inherits from another class, we need to add them to the infill as well
                if item[1] != None:
                    prefix, suffix = separate_script(
                        script_text=original_input_string,
                        word=item[1],
                        line_number=key[0],
                    )
                    candidates[level].append(
                        {
                            "infill": item[1],
                            "line": key[0],
                            "prefix": prefix,
                            "suffix": suffix,
                            "level": level,
                        }
                    )
            elif level in ("strings", "variable_names"):
                prefix, suffix = separate_script(
                    script_text=original_input_string, word=item, line_number=key[0]
                )
                candidates[level].append(
                    {
                        "infill": item,
                        "prefix": prefix,
                        "suffix": suffix,
                        "level": level,
                    }
                )
            elif level == "comments":
                prefix, suffix = separate_script(
                    script_text=original_input_string, word=item, line_number=key[0]
                )
                candidates[level].append(
                    {
                        "infill": item,
                        "prefix": prefix + "#",
                        "suffix": suffix,
                        "level": level,
                    }
                )
            elif level == "docstrings":
                prefix, suffix = separate_script(
                    script_text=original_input_string, word=item, line_number=key[0]
                )
                candidates[level].append(
                    {
                        "infill": item,
                        "prefix": prefix,
                        "suffix": suffix,
                        "level": level,
                    }
                )

    return candidates


def process_input(
    input_code_string_batch: list,
):
    processed_inputs = {
        input_code_string: prepare_input(input_code_string)
        for input_code_string in input_code_string_batch
    }
    return [
        prepare_inputs_for_infill(input_code_string, processed_input)
        for input_code_string, processed_input in processed_inputs.items()
    ]
