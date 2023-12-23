import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "llava-hf/llava-1.5-7b-hf"
quant_path = "llava-1.5-7b-hf-awq"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')