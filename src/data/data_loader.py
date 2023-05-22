import os
import torch
from torch.utils.data import DataLoader, Dataset


class CodeInfillDataset(Dataset):
    """
    A PyTorch dataset for code infilling tasks.

    Args:
        dataset_path (str): The path to the directory containing the dataset files.
        code_language (str): The programming language of the code files in the dataset.

    Attributes:
        dataset_path (str): The path to the directory containing the dataset files.
        code_language (str): The programming language of the code files in the dataset.
        code_suffix (str): The file suffix for code files in the dataset.
    """

    def __init__(self, dataset_path: str, code_language: str) -> None:
        self.dataset_path = dataset_path
        self.code_language = code_language
        self.code_suffix = ".py" if self.code_language == "python" else ".js"

    def __len__(self) -> int:
        """
        Returns the number of code files in the dataset.

        Returns:
            int: The number of code files in the dataset.
        """
        return len(
            [f for f in os.listdir(self.dataset_path) if f.endswith(self.code_suffix)]
        )

    def __getitem__(self, index: int) -> str:
        """
        Returns the input code string at the given index.

        Args:
            index (int): The index of the input code string to return.

        Returns:
            str: The input code string at the given index.
        """
        try:
            file_path = os.path.join(
                self.dataset_path,
                f"the_stack_{self.code_language}_script_{index}{self.code_suffix}",
            )
            input_code_string = open(file_path, "r").read()
            return input_code_string

        except FileNotFoundError:
            return ""
