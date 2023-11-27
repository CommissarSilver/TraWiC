import unittest, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.checker.checker import Checker


class TestChecker(unittest.TestCase):
    def setUp(self):
        self.checker = Checker(os.path.join(os.getcwd(),'src','main.py'))  # assuming you have a test.py file

    def test_prepare_input(self):
        self.checker.prepare_input()
        self.assertIsNotNone(self.checker.processed_input)

    def test_separate_script(self):
        prefix, suffix = self.checker.separate_script(
            self.checker.original_input, "dataset_files", 1
        )
        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)

    def test_prepare_inputs_for_infill(self):
        levels = [
            "function_names",
            "class_names",
            "variable_names",
            "strings",
            "docstrings",
            "comments",
        ]
        for level in levels:
            candidates = self.checker.prepare_inputs_for_infill(level)
            self.assertIsNotNone(candidates)

    def test_check_similarity(self):
        candidate = {
            "infill": "test",
            "prefix": "prefix",
            "suffix": "suffix",
            "level": "level",
        }
        result = self.checker.check_similarity("test", candidate)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
