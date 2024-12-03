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
model = AutoAWQForCausalLM.from_pretrained(model_path)
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
model = AutoAWQForCausalLM.from_pretrained(model_path)
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
model = AutoAWQForCausalLM.from_pretrained(model_path)
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

### Vision-Language Models

AutoAWQ supports a few vision-language models. So far, we support LLaVa 1.5 and LLaVa 1.6 (next).

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'llava-hf/llama3-llava-next-8b-hf'
quant_path = 'llama3-llava-next-8b-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

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
model = AutoAWQForCausalLM.from_pretrained(model_path)
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

### Custom Quantizer (Qwen2 VL Example)

Below, the Qwen team has provided an example of how to use a custom quantizer. This works to
effectively quantize the Qwen2 VL model using multimodal examples.

```python
import torch
import torch.nn as nn

from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

# Specify paths and hyperparameters for quantization
model_path = "Qwen/Qwen2-VL-7B-Instruct"
quant_path = "qwen2-vl-7b-instruct"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path, attn_implementation="flash_attention_2"
)

# We define our own quantizer by extending the AwqQuantizer.
# The main difference is in how the samples are processed when
# the quantization process initialized.
class Qwen2VLAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:  # work with early exit
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

# Then you need to prepare your data for calibaration. What you need to do is just put samples into a list,
# each of which is a typical chat message as shown below. you can specify text and image in `content` field:
# dataset = [
#     # message 0
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me who you are."},
#         {"role": "assistant", "content": "I am a large language model named Qwen..."},
#     ],
#     # message 1
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "file:///path/to/your/image.jpg"},
#                 {"type": "text", "text": "Output all text in the image"},
#             ],
#         },
#         {"role": "assistant", "content": "The text in the image is balabala..."},
#     ],
#     # other messages...
#     ...,
# ]
# here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.
def prepare_dataset(n_sample: int = 8) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]

dataset = prepare_dataset()

# process the dataset into tensors
text = model.processor.apply_chat_template(dataset, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = model.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

# Then just run the calibration process by one line of code
model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)

# Save the model
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
```

### Another Custom Quantizer (MiniCPM3 Example)

Here we introduce another custom quantizer from the MiniCPM team at OpenBMB. We only
modify the weight clipping mechanism to make quantization work.

```python
import torch
from transformers import AutoTokenizer

from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer, clear_memory

class CPM3AwqQuantizer(AwqQuantizer):
    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        if org_w_shape[0] % oc_batch_size != 0:
            oc_batch_size = org_w_shape[0]
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

model_path = 'openbmb/MiniCPM3-4B'
quant_path = 'minicpm3-4b-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, quantizer_cls=CPM3AwqQuantizer)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
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
To run inference with CPU , you should specify `use_ipex=True`. ipex is the backend for CPU including kernel for operators. ipex is intel_extension_for_pytorch package.

```python
from awq import AutoAWQForCausalLM

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, use_ipex=True)
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
import torch
import requests
from PIL import Image
from awq import AutoAWQForCausalLM
from transformers import AutoProcessor, TextStreamer

# Load model
quant_path = "casperhansen/llama3-llava-next-8b-awq"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

# Define prompt
prompt = """\
<|im_start|>system\nAnswer the questions.<|im_end|>
<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>
<|im_start|>assistant
"""

# Define image
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# Load inputs
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

generation_output = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=streamer
)
```

### Qwen2 VL

Below is an example of how to run inference using Qwen2 VL.

```python
from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, TextStreamer

# Load model
quant_path = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Load inputs
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generation_output = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=streamer
)
```