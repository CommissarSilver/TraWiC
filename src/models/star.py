import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_CACHE"] = "/store/travail/vamaj/hf_cache"
checkpoint = "bigcode/starcoder"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

time_for_load_model_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, cache_dir="/store/travail/vamaj/hf_cache"
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, cache_dir="/store/travail/vamaj/hf_cache"
).to(device)
print(
    "\033[92m time for loading model in seconds: ",
    time.time() - time_for_load_model_start,
    "\033[0m",
)

input_1_time = time.time()
inputs_1 = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs_1 = model.generate(inputs_1)
# print(tokenizer.decode(outputs_1[0]))
input_1_finish = time.time()

print(
    "\033[91m time for inference in seconds: ", input_1_finish - input_1_time, "\033[0m"
)

# inputs_2 = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
# outputs_2 = model.generate(inputs_2)
# print(tokenizer.decode(outputs_2[0]))
