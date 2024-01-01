import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'TheBloke/Llama-2-7B-Chat-fp16'
quant_path = 'llama-2-7b-3bit-awq'
llama_cpp_path = '/workspace/llama.cpp'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 3, "version": "GEMM" }

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
    gguf_compatible=True
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')

# GGUF conversion
print('Converting model to GGUF...')

if not os.path.exists(llama_cpp_path):
    cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path} && cd {llama_cpp_path} && make LLAMA_CUBLAS=1"
    subprocess.run([cmd], shell=True, check=True)

# Step 1: python convert.py {quant_path} --outfile {quant_path}/model.gguf
convert_script = "convert.py" if model.model_type == 'llama' else "convert-hf-to-gguf.py"
convert_cmd_path = os.path.join(llama_cpp_path, convert_script)
subprocess.run([
    f"python {convert_cmd_path} {quant_path} --outfile {quant_path}/model.gguf"
], shell=True, check=True)

# Step 2: ./quantize {quant_path}/model.gguf {quant_path}/model_q3.gguf q3_k
llama_cpp_method = f"q{quant_config['w_bit']}_k"
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")
subprocess.run([
    f"{quantize_cmd_path} {quant_path}/model.gguf {quant_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"
], shell=True, check=True)

# Step 3: Inference using llama.cpp
main_cmd_path = os.path.join(llama_cpp_path, "main")
prompt = "[INST] <<SYS>>You are a helpful, respectful and honest assistant.<</SYS>>Hello![/INST]"
cmd = f"""{main_cmd_path} \
  -m {quant_path}/model_{llama_cpp_method}.gguf \
  -p "{prompt}" \
  --repeat_penalty 1 \
  --no-penalize-nl \
  --color --temp 0 -c 512 --n-gpu-layers 33"""
subprocess.run([cmd], shell=True, check=True)