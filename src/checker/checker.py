import ast
import logging
import os
import re
from typing import Dict, List, Tuple, Union

from fuzzywuzzy import fuzz

logger = logging.getLogger("checker")


class Checker:
    """
    A class for checking code files.

    Attributes:
    -----------
    input_path : str
        The path to the input file.
    original_input : str
        The original input file contents.
    """

    def __init__(self, input_path: str) -> None:
        # read the input file
        if input_path.endswith(".py"):
            self.input_path = input_path
            self.original_input = open(self.input_path, "r").read()
        else:  #! For now only python is supported
            raise NotImplementedError
        # extract the items from the input file
        self.prepare_input()

    def prepare_input(self) -> None:
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
        comments_iter = re.finditer(r"\s*#(.*)", self.original_input)
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
            logger.debug(f"Extracted docstrings: {docstrings}")
        except Exception as e:
            logger.exception(f"Error in extracting docstrings: {e}")
            docstrings = None

        try:
            comments = {
                (
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
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
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
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
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
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
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
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
                    self.original_input.count("\n", 0, match.start()) + 1,
                    self.original_input.count("\n", 0, match.end()) + 1,
                ): match.group()
                for match in strings_iter
            }
            logger.debug(f"Extracted strings: {strings}")
        except Exception as e:
            logger.exception(f"Error in extracting strings: {e}")
            strings = None

        logger.info("Finished preparing input")

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
        try:
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

            logger.info("Finished separating script")

            return prefix, suffix
        except Exception as e:
            logger.exception(f"Error in separating script: {e}. Returning empty strings")
            return "", ""

    def prepare_inputs_for_infill(self, level: str) -> List[Dict]:
        """
        Prepares the input for the infill model

        Args:
            level (str): "fuinction_names", "class_names", "variable_names", "strings", "docstrings", "comments"

        Returns:
            List[Dict]: list of candidates for the infill model
        """
        candidates = []

        if self.processed_input[level] != None:
            model_input_candidates = self.processed_input[level]
        else:
            model_input_candidates = {}

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
                        "level": level,
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
                            "level": level,
                        }
                    )
            elif level in ("strings", "variable_names"):
                prefix, suffix = Checker.separate_script(
                    script_text=self.original_input, word=item, line_number=key[0]
                )
                candidates.append(
                    {
                        "infill": item,
                        "prefix": prefix,
                        "suffix": suffix,
                        "level": level,
                    }
                )
            elif level == "comments":
                prefix, suffix = Checker.separate_script(
                    script_text=self.original_input, word=item, line_number=key[0]
                )
                candidates.append(
                    {
                        "infill": item,
                        "prefix": prefix + "#",
                        "suffix": suffix,
                        "level": level,
                    }
                )
            elif level == "docstrings":
                prefix, suffix = Checker.separate_script(
                    script_text=self.original_input, word=item, line_number=key[0]
                )
                candidates.append(
                    {
                        "infill": item,
                        "prefix": prefix,
                        "suffix": suffix,
                        "level": level,
                    }
                )

        return candidates

    @staticmethod
    def check_similarity(
        model_output: str,
        candidate: dict,
        similiarity_metric: str = "exact",
    ) -> int:
        """
        Checks the similarity between the model output and a candidate infill.

        Args:
            model_output (str): output of the model
            candidate (dict): the candidate dictionary
            similiarity_metric (str, optional): which similarity method to select. Defaults to "exact".

        Returns:
            int: results of the similarity check
        """
        if model_output != None:
            if similiarity_metric == "exact":
                if candidate["infill"].upper() == model_output.strip().upper():
                    logger.debug(
                        f"Similarity metric: ( {similiarity_metric} ). found infill objective in model output. infill objective: ( {candidate['infill']} ), model output: ( {model_output} )"
                    )
                    logger.info(f"Found infill objective: {candidate['infill']}")

                    return 1
                else:
                    logger.debug(
                        f"Similarity metric: ( {similiarity_metric} ). didn't find infill objective in model output. infill objective: ({candidate['infill']}), model output: ( {model_output} )"
                    )
                    logger.info(f"Didn't find infill objective: {candidate['infill']}")

                    return 0
            elif similiarity_metric == "fuzzy":
                similarity_ratio = fuzz.ratio(candidate["infill"], model_output)
                logger.debug(
                    f"Similarity metric: ( {similiarity_metric} ). similarity ratio: ( {similarity_ratio} ). infill objective: ( {candidate['infill']} ), model output: ( {model_output} )"
                )
                logger.info(
                    f"Found similarity ratio for {candidate['infill']}: {similarity_ratio}"
                )

                return similarity_ratio
            else:
                raise Exception(
                    f"Similarity metric: ( {similiarity_metric} ) is not supported"
                )
        else:
            logger.exception(
                f"MODEL RETURNED NONE. infill objective: ( {candidate['infill']} )"
            )
            logger.info(
                f"Didn't find infill objective: {candidate['infill']} - Model returned None"
            )

            return 0


class CheckerBlock:
    """
    A class for checking blocks of code files.

    Attributes:
    -----------
    input_path : str
        The path to the input file.
    original_input : str
        The original input file contents."""

    def __init__(self, input_path: str) -> None:
        # read the input file
        if input_path.endswith(".py"):
            self.input_path = input_path
            self.original_input = open(self.input_path, "r").read()
        else:  #! For now only python is supported
            raise NotImplementedError
        # extract the items from the input file
        self.prepare_input()

    def prepare_input(self):
        """
        Extracts the classes and functions from the input file by using the ast module
        """
        classes = []
        functions = []
        try:
            tree = ast.parse(self.original_input)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_body = ast.get_source_segment(
                        self.original_input, node, padded=True
                    )
                    classes.append((class_name, class_body))

                elif isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    function_body = ast.get_source_segment(
                        self.original_input, node, padded=True
                    )
                    functions.append((function_name, function_body))
            logger.info(
                f"Extracted {len(classes)} classes and {len(functions)} functions from {self.input_path} for CheckerBlcok"
            )
        except Exception as e:
            logger.info(
                f"Couldn't extract classes and functions from {self.input_path} for CheckerBlcok - Skipping"
            )
            pass

        self.classes = classes
        self.functions = functions

    def prepare_inputs_for_prediction(self):
        # break down the functions and classes int half with a context token window of 2048
        candidates = []
        for function in self.functions:
            # count the number of lines in the function
            lines_num = len(function[1].split("\n"))
            tokens_num = len(function[1].split(" "))
            if tokens_num > 2048:
                logger.debug(
                    "SantaCoder does not support inputs with more than 2048 tokens. Skipping"
                )
                continue
            # break the function into two parts while keeping the tokens intact so that it doesn't cut words in half
            inputs = (
                "\n".join(function[1].split("\n")[: lines_num // 2]),
                "\n".join(function[1].split("\n")[lines_num // 2 :]),
            )
            candidates.append(
                {
                    "file_path": self.input_path,
                    "prefix": inputs[0],
                    "suffix": inputs[1],
                    "level": "functions",
                }
            )
            logger.info(
                f"Created function candidates for {self.input_path} in CheckerBlock"
            )
        #! For now classes are not supported for investigation
        # for class_ in self.classes:
        #     tokens_num = len(class_[1].split(" "))
        #     if tokens_num > 2048:
        #         continue
        #     inputs = (class_[1][: tokens_num // 2], class_[1][tokens_num // 2 :])
        #     candidates.append(
        #         {
        #             "file_path": self.input_path,
        #             "prefix": inputs[0],
        #             "suffix": inputs[1],
        #             "level": "classes",
        #         }
        #     )
        return candidates

    @staticmethod
    def check_similarity(model_output, candidate):
        #! This will be handled by NiCAD
        pass


if __name__ == "__main__":
    checker = Checker(
        os.path.join(
            os.getcwd(),
            "data",
            "the_stack",
            "python",
            "the_stack_python_script_0.py",
        )
    )
    checker.prepare_input()
    x = checker.prepare_inputs_for_infill("strings")
    print(x)

    checker_block = CheckerBlock(
        os.path.join(
            os.getcwd(),
            "data",
            "the_stack",
            "python",
            "the_stack_python_script_32845.py",
        )
    )
    x = checker_block.prepare_inputs_for_prediction()
    print(x)
