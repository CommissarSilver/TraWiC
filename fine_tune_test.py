import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

adapter_id = os.path.join("/home/vamaj/scratch/TraWiC/llms/mistral_fim")

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
device_map = {"": 0}
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/vamaj/scratch/TraWiC/llms/mistral",
    device_map=device_map,
    local_files_only=True,
)
new_model = PeftModel.from_pretrained(
    base_model,
    adapter_id,
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/vamaj/scratch/TraWiC/llms/mistral",
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

num_added_special_tokens = tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            FIM_PREFIX,
            FIM_MIDDLE,
            FIM_SUFFIX,
            FIM_PAD,
        ],
    }
)

new_model.resize_token_embeddings(len(tokenizer))

input_text = "<fim-prefix>def<fim-suffix>print(message)<fim-middle>"
encoded_input = tokenizer.encode(input_text, return_tensors="pt")
with torch.no_grad():
    outs = new_model.generate(input_ids=encoded_input.to("cuda"))
    if outs.dim() > 1:
        outs = outs[0]

    print(tokenizer.decode(outs, skip_special_tokens=True))
