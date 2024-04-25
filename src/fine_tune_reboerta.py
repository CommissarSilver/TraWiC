from transformers import DataCollatorForLanguageModeling
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from models import Roberta
import os
from torch.utils.data import Dataset
import torch

roberta = Roberta()
num_epochs = 100


def load_scripts_from_directory(directory_path):
    script_texts = []

    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as file:
                    script_texts.append(file.read())
                    if len(script_texts) > 20:
                        return script_texts


# Load all python files from the 'data' directory
training_texts = load_scripts_from_directory("/store/travail/vamaj/TWMC/data")

tokenized_inputs = roberta.tokenizer(
    training_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
)


class PythonScriptsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = PythonScriptsDataset(tokenized_inputs)


roberta.model.train()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=roberta.tokenizer, mlm=True, mlm_probability=0.15
)

train_dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, collate_fn=data_collator
)

optimizer = AdamW(roberta.model.parameters(), lr=5e-5)
num_train_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * num_train_steps,
    num_training_steps=num_train_steps,
)


for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs = batch["input_ids"].to(roberta.device)
        labels = batch["labels"].to(roberta.device)

        optimizer.zero_grad()
        outputs = roberta.model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Training Loss: {loss.item()}")
