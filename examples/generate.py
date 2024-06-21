from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer


quant_path = "casperhansen/llama-3-8b-instruct-awq"

# Load model
if get_best_device() == "cpu":
    model = AutoAWQForCausalLM.from_quantized(quant_path, use_qbits=True, fuse_layers=False)
else:
    model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

chat = [
    {"role": "system", "content": "You are a concise assistant that helps answer questions."},
    {"role": "user", "content": prompt},
]

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

tokens = tokenizer.apply_chat_template(
    chat,
    return_tensors="pt"
)
tokens = tokens.to(get_best_device())

# Generate output
generation_output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=64,
    eos_token_id=terminators
)
