import os

from torch.utils.data import DataLoader

from data.data_loader import CodeInfillDataset
from checker.colator import process_input

j = CodeInfillDataset(
    dataset_path=os.path.join(os.getcwd(), "data", "the_stack", "python"),
    code_language="python",
)
train_dataloader = DataLoader(
    j,
    collate_fn=process_input,
    batch_size=64,
    shuffle=True,
)
for i, x in enumerate(train_dataloader):
    print(i)
