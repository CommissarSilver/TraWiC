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
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer
import argparse

#### code adapted from: https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html

parser = argparse.ArgumentParser(description="Argument Parser for Model Configuration")

# Model Paths
parser.add_argument(
    "--model_name",
    type=str,
    default=os.path.join("/home/vamaj/scratch/TraWiC/llms/llama"),
    help="Path to the model",
)

parser.add_argument(
    "--new_model",
    type=str,
    default="/home/vamaj/scratch/TraWiC/llms/llama_fim",
    help="Path to the fine-tuned model",
)

# QLoRA Parameters
parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension")
parser.add_argument(
    "--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling"
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.1,
    help="Dropout probability for LoRA layers",
)

# bitsandbytes Parameters
parser.add_argument(
    "--use_4bit",
    type=bool,
    default=True,
    help="Activate 4-bit precision base model loading",
)
parser.add_argument(
    "--bnb_4bit_compute_dtype",
    type=str,
    default="float16",
    help="Compute dtype for 4-bit base models",
)
parser.add_argument(
    "--bnb_4bit_quant_type",
    type=str,
    default="nf4",
    help="Quantization type (fp4 or nf4)",
)
parser.add_argument(
    "--use_nested_quant",
    type=bool,
    default=False,
    help="Activate nested quantization for 4-bit base models (double quantization)",
)

# TrainingArguments Parameters
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results",
    help="Output directory where the model predictions and checkpoints will be stored",
)
parser.add_argument(
    "--num_train_epochs", type=int, default=1, help="Number of training epochs"
)
parser.add_argument(
    "--fp16",
    type=bool,
    default=False,
    help="Enable fp16 training (set bf16 to True with an A100)",
)
parser.add_argument("--bf16", type=bool, default=True, help="Enable bf16 training")
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=4,
    help="Batch size per GPU for training",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=4,
    help="Batch size per GPU for evaluation",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of update steps to accumulate the gradients for",
)
parser.add_argument(
    "--gradient_checkpointing",
    type=bool,
    default=True,
    help="Enable gradient checkpointing",
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=0.3,
    help="Maximum gradient normal (gradient clipping)",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-4,
    help="Initial learning rate (AdamW optimizer)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.001,
    help="Weight decay to apply to all layers except bias/LayerNorm weights",
)
parser.add_argument(
    "--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use"
)
parser.add_argument(
    "--lr_scheduler_type", type=str, default="constant", help="Learning rate schedule"
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Number of training steps (overrides num_train_epochs)",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.03,
    help="Ratio of steps for a linear warmup",
)
parser.add_argument(
    "--group_by_length",
    type=bool,
    default=True,
    help="Group sequences into batches with same length",
)
parser.add_argument(
    "--save_steps", type=int, default=25, help="Save checkpoint every X updates steps"
)
parser.add_argument(
    "--logging_steps", type=int, default=25, help="Log every X updates steps"
)

# SFT Parameters
parser.add_argument(
    "--max_seq_length", type=int, default=None, help="Maximum sequence length to use"
)
parser.add_argument(
    "--packing",
    type=bool,
    default=False,
    help="Pack multiple short examples in the same input sequence to increase efficiency",
)

# Load the entire model on GPU 0
parser.add_argument(
    "--device_map",
    type=dict,
    default={"": 0},
    help="Device map to load the entire model on GPU",
)

# Parse arguments
args = parser.parse_args()


def get_jsons_list(jsons_dir):
    jsons_list = []
    for root, dirs, files in os.walk(jsons_dir):
        for file in files:
            if file.endswith(".json"):
                jsons_list.append(os.path.join(root, file))
    return jsons_list


def json_to_prompt(prefix, suffix, infill):
    prompt = (
        f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>{infill}<|endoftext|>"
    )
    return prompt


datafiles = get_jsons_list("/home/vamaj/scratch/TraWiC/data/finetune_ds")
print(datafiles)
dataset = load_dataset(
    "json",
    data_files=datafiles,
)
dataset = dataset.map(
    lambda x: {"text": json_to_prompt(x["prefix"], x["suffix"], x["infill"])}
)

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and args.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map=args.device_map,
    local_files_only=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
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
        "additional_special_tokens": [FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
    }
)

model.resize_token_embeddings(len(tokenizer))


def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split(".")[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


if os.path.exists(args.new_model):
    peft_config = PeftConfig.from_pretrained(args.new_model)
    model = PeftModel.from_pretrained(
        model,
        args.new_model,
        is_trainable=True
    )
    print("merged model")
    model.print_trainable_parameters()

else:
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_target_modules(model),
    )


# Set training parameters
training_arguments = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    fp16=args.fp16,
    bf16=args.bf16,
    max_grad_norm=args.max_grad_norm,
    max_steps=args.max_steps,
    warmup_ratio=args.warmup_ratio,
    group_by_length=args.group_by_length,
    lr_scheduler_type=args.lr_scheduler_type,
    report_to="tensorboard",
)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=args.packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(args.new_model)
