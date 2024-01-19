from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "TheBloke/TinyLlama-1.1B-python-v0.1-AWQ"

# Load model
tokenizer = AutoTokenizer.from_pretrained(quant_path)
# Generate output
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")

# Load model
print("Loading GEMM model...")
gemm_model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
gemm_out = gemm_model.generate(inp, max_new_tokens=100)

print("GEMM output:")
print(tokenizer.decode(gemm_out[0]))
print()

print("Loading Exllama model...")
exllama_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama=True
)
exllama_out = exllama_model.generate(inp, max_new_tokens=100)

print("Exllama output:")
print(tokenizer.decode(exllama_out[0]))


print("Loading ExllamaV2 model...")
exllama_v2_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama_v2=True
)
exllama_v2_out = exllama_v2_model.generate(inp, max_new_tokens=100)

print("ExllamaV2 output:")
print(tokenizer.decode(exllama_v2_out[0]))
