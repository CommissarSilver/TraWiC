import os, re, sys, logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import logger_utils

logger = logger_utils.CustomLogger(
    "checker.log", create_directory=False, log_level=logging.INFO
)


class Checker:
    def __init__(self, input_path: str) -> None:
        if input_path.endswith(".py"):
            self.input_path = input_path
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
        input = open(self.input_path, "r").read()

        # use regex to extract the mentioned items
        docstrings_iter = re.finditer(r'"""[\s\S]*?"""', input)
        comments_iter = re.finditer(r"#.*", input)
        function_names_iter = re.finditer(
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", input
        )
        class_names_iter = re.finditer(
            r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?", input
        )
        variable_names_iter = re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", input)
        strings_iter = re.finditer(r"\".*?\"|'.*?'", input)

        # for each of the items, we need to store their value and their line numbers
        try:
            docstrings = {
                (
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
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
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
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
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
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
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
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
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
                ): match.group()
                for match in variable_names_iter
            }
            logger.debug(f"variable_names: {variable_names}")
        except Exception as e:
            logger.exception(f"error in extracting variable names: {e}")
            variable_names = None

        try:
            strings = {
                (
                    input.count("\n", 0, match.start()) + 1,
                    input.count("\n", 0, match.end()) + 1,
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


if __name__ == "__main__":
    checker = Checker(
        "/Users/ahura/Nexus/TWMC/data/the_stack/python/the_stack_python_script_0.py"
    )
    checker.prepare_input()
    print(checker.processed_input)
