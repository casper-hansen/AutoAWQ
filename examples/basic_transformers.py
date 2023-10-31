import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# NOTE: Must install from PR until merged
# pip install --upgrade git+https://github.com/younesbelkada/transformers.git@add-awq
model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
text = "[INST] What are the basic steps to use the Huggingface transformers library? [/INST]"

tokens = tokenizer(
    text, 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)