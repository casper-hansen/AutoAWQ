import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'bigscience/bloom-1b1'
quant_path = 'bloom-1b1'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, load_safetensors=True, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
