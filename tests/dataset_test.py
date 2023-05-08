import unittest, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import dataset


class TestTheStack(unittest.TestCase):
    def test_getting_the_dataset(self):
        dataset.get_thestack_dataset(scripts_num=10)
        

    def test_directory_creation(self):
        dataset.get_thestack_dataset(scripts_num=10)

        # Assert that the directory is created
        self.assertTrue(
            os.path.exists(os.path.join(os.getcwd(), "data", "the_stack", "python"))
        )


if __name__ == "__main__":
    import os

    env_vars = os.environ
    print(env_vars)
