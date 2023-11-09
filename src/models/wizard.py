import torch, inspect, logging
from typing import Tuple

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-15B-V1.0")
model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-15B-V1.0").to("cuda")
instruction = "predict the MASKED token def MASKED():\n\tprint('hello world')"
inputs = tokenizer.encode(
    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:"
)
outputs = model.generate(inputs)
