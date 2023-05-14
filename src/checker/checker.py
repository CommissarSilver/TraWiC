import os, re, sys, logging
from typing import Tuple, List, Dict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import logger_utils

logger = logger_utils.CustomLogger(
    "checker.log", create_directory=True, log_level=logging.DEBUG
)


class Checker:
    def __init__(self, input_path: str) -> None:
        if input_path.endswith(".py"):
            self.input_path = input_path
            self.original_input = open(self.input_path, "r").read()
        else:
            raise NotImplementedError
        self.processed_input = None

    def prepare_input(self):
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
        docstrings_iter = re.finditer(r'"""[\s\S]*?"""', self.original_input)
        comments_iter = re.finditer(r"#.*", self.original_input)
        function_names_iter = re.finditer(
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", self.original_input
        )
        class_names_iter = re.finditer(
            r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?", self.original_input
        )
        variable_names_iter = re.finditer(
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", self.original_input
        )
        strings_iter = re.finditer(r"\".*?\"|'.*?'", self.original_input)

        # for each of the items, we need to store their value and their line numbers
        try:
            docstrings = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): match.group()
                for match in docstrings_iter
            }
            logger.debug(f"docstrings: {docstrings}")
        except Exception as e:
            logger.exception(f"error in extracting docstrings: {e}")
            docstrings = None

        try:
            comments = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): match.group()
                for match in comments_iter
            }
            logger.debug(f"comments: {comments}")
        except Exception as e:
            logger.exception(f"error in extracting comments: {e}")
            comments = None

        try:
            function_names = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): (match.group(1), match.group(2))
                for match in function_names_iter
            }
            logger.debug(f"function_names: {function_names}")
        except Exception as e:
            logger.exception(f"error in extracting function names:{e}")
            function_names = None

        try:
            class_names = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): (match.group(1), match.group(2))
                for match in class_names_iter
            }
            logger.debug(f"class_names: {class_names}")
        except Exception as e:
            logger.exception(f"error in extracting class names: {e}")
            class_names = None

        try:
            variable_names = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): match.group(1)
                for match in variable_names_iter
            }
            logger.debug(f"variable_names: {variable_names}")
        except Exception as e:
            logger.exception(f"error in extracting variable names: {e}")
            variable_names = None

        try:
            strings = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): match.group()
                for match in strings_iter
            }
            logger.debug(f"strings: {strings}")
        except Exception as e:
            logger.exception(f"error in extracting strings: {e}")
            strings = None

        logger.info("finished preparing input")

        self.processed_input = {
            "docstrings": docstrings,
            "comments": comments,
            "function_names": function_names,
            "class_names": class_names,
            "variable_names": variable_names,
            "strings": strings,
        }

    @staticmethod
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

    def prepare_inputs_for_infill(self, level: str) -> List[Dict[str, str, str, str]]:
        """
        Prepares the input for the infill model

        Args:
            level (str): "fuinction_names", "class_names", "variable_names", "strings", "docstrings", "comments"

        Returns:
            List[Dict[str, str, str, str]]: list of candidates for the infill model
        """
        candidates = []

        if self.processed_input[level] != None:
            model_input_candidates = self.processed_input[level]
        else:
            raise "there are no candidates for this level"

        for key, item in model_input_candidates.items():
            if level in ("function_names", "class_names"):
                # for function and class names, we need to separate the script into prefix and suffix
                prefix, suffix = Checker.separate_script(
                    script_text=self.original_input, word=item[0], line_number=key[0]
                )
                candidates.append(
                    {
                        "infill": item[0],
                        "line": key[0],
                        "prefix": prefix,
                        "suffix": suffix,
                    }
                )
                # if the function has arguments, or the class inherits from another class, we need to add them to the infill as well
                if item[1] != None:
                    prefix, suffix = Checker.separate_script(
                        script_text=self.original_input, word=item[1], line_number=key[0]
                    )
                    candidates.append(
                        {
                            "infill": item[1],
                            "line": key[0],
                            "prefix": prefix,
                            "suffix": suffix,
                        }
                    )
            elif level in ("strings", "variable_names"):
                prefix, suffix = Checker.separate_script(
                    script_text=self.original_input, word=item, line_number=key[0]
                )
                candidates.append({"infill": item, "prefix": prefix, "suffix": suffix})
            elif level in ("docstrings", "comments"):
                raise "not implemented yet"
        
        return candidates


if __name__ == "__main__":
    checker = Checker(
        "/Users/ahura/Nexus/TWMC/data/the_stack/python/the_stack_python_script_0.py"
    )
    checker.prepare_input()
    x = checker.prepare_inputs_for_infill("strings")
    print(x)
