from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/Llama-2-7B-Chat-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
[INST] <<SYS>>
You are a helpful, respectful and honest assistant that answers questions concisely.
<</SYS>>
{prompt}[/INST]"""

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)