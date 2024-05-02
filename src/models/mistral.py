# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.model import InfillModel
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
device = "cpu"

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

num_added_special_tokens = tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
    }
)

model.resize_token_embeddings(len(tokenizer))

input_text = "<fim-prefix>def fib(n):<fim-suffix>    else:\n        return fib(n - 2) + fib(n - 1)<fim-middle>"
inputs = tokenizer.encode(
    input_text,
    return_tensors="pt",
    return_token_type_ids=False,
).to(device)
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

class MistralCoder(InfillModel)