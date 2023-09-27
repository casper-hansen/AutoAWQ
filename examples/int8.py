from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'PY007/TinyLlama-1.1B-Chat-v0.2'
quant_path = 'tinyllama-chat-awq'
quant_config = { "zero_point": False, "q_group_size": 0, "w_bit": 8, "version": "SmoothQuant" }

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(model_path, **dict(low_cpu_mem_usage=True))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
# NOTE: pass safetensors=True to save quantized model weights as safetensors
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')