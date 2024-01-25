from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

system = "You are a helpful assistant that answers precisely."

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(system=system, prompt=prompt), 
    return_tensors='pt'
).input_ids.to("mps")

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=64
)