import os
import torch
from torch.utils.data import DataLoader, Dataset


class CodeInfillDataset(Dataset):
    def __init__(self, dataset_path, code_language) -> None:
        self.dataset_path = dataset_path
        self.code_language = code_language
        self.code_suffix = ".py" if self.code_language == "python" else ".js"

    def __len__(self):
        return len(
            [f for f in os.listdir(self.dataset_path) if f.endswith(self.code_suffix)]
        )

    def __getitem__(self, index):
        try:
            file_path = os.path.join(
                self.dataset_path,
                f"the_stack_{self.code_language}_script_{index}{self.code_suffix}",
            )
            input_code_string = open(file_path, "r").read()
            return input_code_string
        except FileNotFoundError:
            pass
