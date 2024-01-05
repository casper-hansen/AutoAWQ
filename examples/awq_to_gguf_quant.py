import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-v0.1'
quant_path = 'mistral-awq'
llama_cpp_path = '/workspace/llama.cpp'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 6, "version": "GEMM" }

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
# NOTE: We avoid packing weights, so you cannot use this model in AutoAWQ
# after quantizing. The saved model is FP16 but has the AWQ scales applied.
model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')

# GGUF conversion
print('Converting model to GGUF...')
llama_cpp_method = "q4_K_M"
convert_cmd_path = os.path.join(llama_cpp_path, "convert.py")
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")

if not os.path.exists(llama_cpp_path):
    cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path} && cd {llama_cpp_path} && make LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1"
    subprocess.run([cmd], shell=True, check=True)

subprocess.run([
    f"python {convert_cmd_path} {quant_path} --outfile {quant_path}/model.gguf"
], shell=True, check=True)

subprocess.run([
    f"{quantize_cmd_path} {quant_path}/model.gguf {quant_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"
], shell=True, check=True)
