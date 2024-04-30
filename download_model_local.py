from huggingface_hub import snapshot_download
import os
# download Mistral
snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.1",
    local_dir= os.path.join(os.getcwd(),'mistral')
    )
# download Llama
snapshot_download(
    repo_id="meta-llama/Llama-2-7b",
    local_dir= os.path.join(os.getcwd(),'llama')
    )