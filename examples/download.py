from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='TitanML/llama2-7b-base-4bit-AWQ', #'TitanML/llama2-13b-base-4bit-AWQ'
    local_dir = '/home/titan-6/AutoAWQ/examples/models/llama-7b-awq',
    local_dir_use_symlinks=False
)
