from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'openai/whisper-large-v3'
quant_path = 'whisper-large-awq'
modules_to_not_convert = ["encoder_attn", "encoder"]

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, modules_to_not_convert=modules_to_not_convert)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')