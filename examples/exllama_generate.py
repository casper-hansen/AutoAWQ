from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "TheBloke/TinyLlama-1.1B-python-v0.1-AWQ"

tokenizer = AutoTokenizer.from_pretrained(quant_path)

print("Loading GEMM model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
gemm_model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)

gemm_logits = gemm_model(inp).logits
gemm_ids = gemm_model.generate(inp, max_new_tokens=20)

print("GEMM output:")
print("=====================================")
print(tokenizer.decode(gemm_ids[0]))
print("=====================================")

print("Loading Exllama model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
exllama_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama=True
)

exllama_logits = exllama_model(inp).logits
print("Max difference with GEMM:", (gemm_logits - exllama_logits).abs().max())
print("Mean difference with GEMM:", (gemm_logits - exllama_logits).abs().mean())


print("Exllama output:")
print("=====================================")
exllama_ids = exllama_model.generate(inp, max_new_tokens=20)
print(tokenizer.decode(exllama_ids[0]))
print("=====================================")


print("Loading ExllamaV2 model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
exllama_v2_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama_v2=True
)

exllama_v2_logits = exllama_v2_model(inp).logits
print("Max difference with GEMM:", (gemm_logits - exllama_v2_logits).abs().max())
print("Mean difference with GEMM:", (gemm_logits - exllama_v2_logits).abs().mean())

print("ExllamaV2 output:")
print("=====================================")
exllama_v2_ids = exllama_v2_model.generate(inp, max_new_tokens=20)
print(tokenizer.decode(exllama_v2_ids[0]))
print("=====================================")
