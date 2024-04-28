import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


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


datafiles = get_jsons_list("/Users/ahvra/Nexus/TWMC/data/finetune_ds")

ds = load_dataset("json", data_files=datafiles)
ds = ds.map(lambda x: {"text": json_to_prompt(x["prefix"], x["suffix"], x["infill"])})
for i in ds["train"]:
    j = i["text"]
    print("hi")
