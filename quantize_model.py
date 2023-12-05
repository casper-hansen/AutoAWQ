from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

model_path = '/root/workspace/external_data/model/base'
quant_path = '/root/workspace/external_data/model/quant'
assert os.path.exists(quant_path)
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMV"
}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    use_flash_attention_2=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=('custom',
                '/root/workspace/external_data/model/calib_data.json'))

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
