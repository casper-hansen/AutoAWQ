from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_dolly():
    data = load_dataset('databricks/databricks-dolly-15k', split="train")

    # concatenate data
    def concatenate_data(x):
        return {"text": x['instruction'] + '\n' + x['context'] + '\n' + x['response']}
    
    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]]

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_dolly())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')