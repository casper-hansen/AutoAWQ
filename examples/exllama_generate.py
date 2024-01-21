import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
tokenizer = AutoTokenizer.from_pretrained(quant_path)


def report_diff(awq, other):
    reldiff = (awq - other).abs() / (awq.abs() + 1e-15)
    print("p90 reldiff with AWQ", torch.quantile(reldiff, 0.9))
    print("Median reldiff with AWQ", reldiff.median())
    print("Mean reldiff with AWQ", reldiff.mean())


print("Model: ", quant_path)
print("Loading GEMM model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
gemm_model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
gemm_logits = gemm_model(inp).logits
gemm_ids = gemm_model.generate(inp, max_new_tokens=50)


# print("GEMM output:")
# print("=====================================")
# print(tokenizer.decode(gemm_ids[0]))
# print("=====================================")


print("Loading Exllama model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
exllama_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama=True
)
exllama_logits = exllama_model(inp).logits
report_diff(gemm_logits, exllama_logits)


# print("Exllama output:")
# print("=====================================")
# exllama_ids = exllama_model.generate(inp, max_new_tokens=50)
# print(tokenizer.decode(exllama_ids[0]))
# print("=====================================")


print("Loading ExllamaV2 model...")
inp = tokenizer("def hello_", return_tensors="pt").input_ids.to("cuda")
exllama_v2_model = AutoAWQForCausalLM.from_quantized(
    quant_path, fuse_layers=False, use_exllama_v2=True
)
exllama_v2_logits = exllama_v2_model(inp).logits
report_diff(gemm_logits, exllama_v2_logits)


# print("ExllamaV2 output:")
# print("=====================================")
# exllama_v2_ids = exllama_v2_model.generate(inp, max_new_tokens=50)
# print(tokenizer.decode(exllama_v2_ids[0]))
# print("=====================================")
