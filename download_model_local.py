from huggingface_hub import snapshot_download
import os

# Path to store the models
base_dir = os.getcwd()

# # Download Mistral with a unique cache directory
# mistral_path = snapshot_download(
#     repo_id="mistralai/Mistral-7B-v0.1",
#     cache_dir=os.path.join(base_dir, "llms", "cache_mistral"),
#     local_dir=os.path.join(base_dir, "llms", "mistral"),
# )

# Download Llama with a unique cache directory
llama_path = snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    cache_dir=os.path.join(base_dir, "llms", "cache_llama"),
    local_dir=os.path.join(base_dir, "llms", "llama"),
)
