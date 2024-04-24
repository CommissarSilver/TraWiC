# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
        "pad_token": EOD,
    }
)

device = "cpu"
model.resize_token_embeddings(len(tokenizer))
input_text = "<fim-prefix>def fib(n):<fim-suffix>    else:\n        return fib(n - 2) + fib(n - 1)<fim-middle>"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
outputs = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
