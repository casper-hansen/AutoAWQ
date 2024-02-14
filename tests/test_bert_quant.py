from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'BAAI/bge-small-en-v1.5'
quant_path = 'bge-small-en-v1.5-quant'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Quantize
model.quantize(tokenizer, quant_config=quant_config, )

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)