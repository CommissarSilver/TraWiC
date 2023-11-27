import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data import dataset


class TestTheStack(unittest.TestCase):
    def test_getting_the_dataset(self):
        dataset.get_thestack_dataset(scripts_num=10)

        # Assert that the directory is created
        self.assertTrue(
            os.path.exists(os.path.join(os.getcwd(), "data", "the_stack", "python"))
        )

    def test_dataset_structure(self):
        # Check if the dataset structure is correct
        dataset_path = os.path.join(os.getcwd(), "data", "the_stack", "python")
        self.assertTrue(os.path.exists(dataset_path))
        self.assertTrue(os.path.isdir(dataset_path))


if __name__ == "__main__":
    unittest.main()
