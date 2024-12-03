import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

device = get_best_device()
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

model = AutoAWQForCausalLM.from_quantized(
  model_id,
  torch_dtype=torch.float16,
  low_cpu_mem_usage=True,
  device_map="auto",
)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": \
        "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"},
]
inputs = tokenizer.apply_chat_template(
  prompt,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt",
  return_dict=True,
).to(device)

outputs = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=256,
    streamer=streamer,
)
