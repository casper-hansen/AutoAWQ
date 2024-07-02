# Examples

## Basic Quantization

AWQ performs zero point quantization down to a precision of 4-bit integers.
You can also specify other bit rates like 3-bit, but some of these options may lack kernels
for running inference.

Notes:

- Some models like Falcon is only compatible with group size 64.
- To use Marlin, you must specify zero point as False and version as Marlin.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

### Custom Data

This includes an example function that loads either wikitext or dolly.
Note that currently all samples above 512 in length are discarded.

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Define data loading methods
def load_dolly():
    data = load_dataset('databricks/databricks-dolly-15k', split="train")

    # concatenate data
    def concatenate_data(x):
        return {"text": x['instruction'] + '\n' + x['context'] + '\n' + x['response']}
    
    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]]

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

#### Long-context: Optimizing quantization

For this example, we will use HuggingFaceTB/cosmopedia-100k as it's a high-quality dataset and
we can filter directly on the number of tokens. We will use Qwen2 7B, one of the newer supported
models in AutoAWQ which is high-performing. The following example ran smoothly on a machine with
an RTX 4090 24 GB VRAM with 107 GB system RAM.

NOTE: Adjusting `n_parallel_calib_samples`, `max_calib_samples`, and `max_calib_seq_len` will help
avoid OOM when customizing your dataset.

- The AWQ algorithm is incredibly sample efficient, so `max_calib_samples` of 128-256 should be
sufficient to quantize a model. A higher number of samples may not be possible without significant
memory available or without further optimizing AWQ with a PR for disk offload.
- When `n_parallel_calib_samples` is set to an integer, we offload to system RAM to save GPU VRAM.
This may cause OOM on your system if you have little memory available; we are looking to optimize
this further in future versions.

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2-7B-Instruct'
quant_path = 'qwen2-7b-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_cosmopedia():
    data = load_dataset('HuggingFaceTB/cosmopedia-100k', split="train")
    data = data.filter(lambda x: x["text_token_length"] >= 2048)

    return [text for text in data["text"]]

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_cosmopedia(),
    n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=4096
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

#### Coding models

For this example, we will use deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct as it's an excellent coding model.

```python
from tqdm import tqdm
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
quant_path = 'deepseek-coder-v2-lite-instruct-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_openhermes_coding():
    data = load_dataset("alvarobartt/openhermes-preferences-coding", split="train")

    samples = []
    for sample in data:
        responses = [f'{response["role"]}: {response["content"]}' for response in sample["chosen"]]
        samples.append("\n".join(responses))

    return samples

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_openhermes_coding(),
    # MODIFY these parameters if need be:
    # n_parallel_calib_samples=32,
    # max_calib_samples=128,
    # max_calib_seq_len=4096
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

### GGUF Export

This computes AWQ scales and appliesthem to the model without running real quantization.
This keeps the quality of AWQ because theweights are applied but skips quantization
in order to make it compatible with other frameworks.

Step by step:

- `quantize()`: Compute AWQ scales and apply them
- `save_pretrained()`: Saves a non-quantized model in FP16
- `convert.py`: Convert the Huggingface FP16 weights to GGUF FP16 weights
- `quantize`: Run GGUF quantization to get real quantized weights, in this case 4-bit.

```python
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
```

## Basic Inference

### Inference With GPU
To run inference, you often want to run with `fuse_layers=True` to get the claimed speedup in AutoAWQ.
Additionally, consider setting `max_seq_len` (default: 2048) as this will be the maximum context that the model can hold.

Notes:

- You can specify `use_exllama_v2=True` to enable ExLlamaV2 kernels during inference.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = "[INST] {prompt} [/INST]"

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)
```

### Inference With CPU
To run inference with CPU , you should specify `use_qbits=True`. QBits is the backend for CPU including kernel for operators. QBits is a module of the intel-extension-for-transformers package. Up to now, the feature of fusing layers hasn't been ready, you should run model with `fuse_layers=False`.

```python
from awq import AutoAWQForCausalLM

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False, use_qbits=True)
```

### Transformers

You can also load an AWQ model by using AutoModelForCausalLM, just make sure you have AutoAWQ installed.
Note that not all models will have fused modules when loading from transformers.
See more [documentation here](https://huggingface.co/docs/transformers/main/en/quantization/awq).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# NOTE: Must install from PR until merged
# pip install --upgrade git+https://github.com/younesbelkada/transformers.git@add-awq
model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
text = "[INST] What are the basic steps to use the Huggingface transformers library? [/INST]"

tokens = tokenizer(
    text, 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)
```

### vLLM

You can also load AWQ models in [vLLM](https://github.com/vllm-project/vllm).

```python
import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

model_path = "casperhansen/mixtral-instruct-awq"

# prompting
prompt = "You're standing on the surface of the Earth. "\
         "You walk one mile south, one mile west and one mile north. "\
         "You end up exactly where you started. Where are you?",

prompt_template = "[INST] {prompt} [/INST]"

# sampling params
sampling_params = SamplingParams(
    repetition_penalty=1.1,
    temperature=0.8,
    max_tokens=512
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# async engine args for streaming
engine_args = AsyncEngineArgs(
    model=model_path,
    quantization="awq",
    dtype="float16",
    max_model_len=512,
    enforce_eager=True,
    disable_log_requests=True,
    disable_log_stats=True,
)

async def generate(model: AsyncLLMEngine, tokenizer: PreTrainedTokenizer):
    tokens = tokenizer(prompt_template.format(prompt=prompt)).input_ids

    outputs = model.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=1,
        prompt_token_ids=tokens,
    )

    print("\n** Starting generation!\n")
    last_index = 0

    async for output in outputs:
        print(output.outputs[0].text[last_index:], end="", flush=True)
        last_index = len(output.outputs[0].text)
    
    print("\n\n** Finished generation!\n")

if __name__ == '__main__':
    model = AsyncLLMEngine.from_engine_args(engine_args)
    asyncio.run(generate(model, tokenizer))
```

### LLaVa (multimodal)

AutoAWQ also supports the LLaVa model. You simply need to load an 
AutoProcessor to process the prompt and image to generate inputs for the AWQ model.

```python
import requests
import torch
from PIL import Image

from awq import AutoAWQForCausalLM
from transformers import AutoProcessor

quant_path = "ybelkada/llava-1.5-7b-hf-awq"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, safetensors=True, device_map={"": 0})
processor = AutoProcessor.from_pretrained(quant_path)

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
# Generate output
generation_output = model.generate(
    **inputs, 
    max_new_tokens=512
)

print(processor.decode(generation_output[0], skip_special_tokens=True))
```
