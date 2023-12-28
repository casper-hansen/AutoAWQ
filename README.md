# AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |

</p>
<p align="center">
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_1000+_models_available-8A2BE2">
    </a>
    <a href="https://github.com/casper-hansen/AutoAWQ/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/release/casper-hansen/AutoAWQ.svg">
    </a>
    <a href="https://pypi.org/project/autoawq/">
        <img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/autoawq/month">
    </a>
</p>

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 3x and reduces memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.  AutoAWQ was created and improved upon from the [original work](https://github.com/mit-han-lab/llm-awq) from MIT.

*Latest News* 🔥
- [2023/12] Mixtral, LLaVa, QWen, Baichuan model support.
- [2023/11] AutoAWQ inference has been integrated into 🤗 transformers. Now includes CUDA 12.1 wheels.
- [2023/10] Mistral (Fused Modules), Bigcode, Turing support, Memory Bug Fix (Saves 2GB VRAM)
- [2023/09] 1.6x-2.5x speed boost on fused models (now including MPT and Falcon).
- [2023/09] Multi-GPU support, bug fixes, and better benchmark scripts available
- [2023/08] PyPi package released and AutoModel class available

## Install

### Prerequisites

- Your GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported.
- Your CUDA version must be CUDA 11.8 or later.
- Requires installing [AutoAWQ kernels](https://github.com/casper-hansen/AutoAWQ_kernels).

### Install from PyPi

To install the newest AutoAWQ from PyPi, you need CUDA 12.1 installed.

```
pip install autoawq
```

If you cannot use CUDA 12.1, you can still use CUDA 11.8 and install the wheel from the [latest release](https://github.com/casper-hansen/AutoAWQ/releases).

```
pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.6/autoawq-0.1.6+cu118-cp310-cp310-linux_x86_64.whl
```

### Build from source

```
git clone https://github.com/casper-hansen/AutoAWQ
cd AutoAWQ
pip install -e .
```

## Supported models

The detailed support list:

| Models   | Sizes                       |
| ---------| ----------------------------|
| LLaMA-2  | 7B/13B/70B                  |
| LLaMA    | 7B/13B/30B/65B              |
| Mistral  | 7B                          |
| Vicuna   | 7B/13B                      |
| MPT      | 7B/30B                      |
| Falcon   | 7B/40B                      |
| OPT      | 125m/1.3B/2.7B/6.7B/13B/30B |
| Bloom    | 560m/3B/7B/                 |
| GPTJ     | 6.7B                        |
| Aquila   | 7B                          |
| Aquila2  | 7B/34B                      |
| Yi       | 6B/34B                      |
| Qwen     | 1.8B/7B/14B/72B             |
| BigCode  | 1B/7B/15B                   |
| GPT NeoX | 20B                         |
| GPT-J    | 6B                          |
| LLaVa    | 7B/13B                      |
| Mixtral  | 8x7B                        |
| Baichuan | 7B/13B                      |
| QWen     | 1.8B/7B/14/72B              |

## Usage

Under examples, you can find examples of how to quantize, run inference, and benchmark AutoAWQ models.

### INT4 GEMM vs INT4 GEMV vs FP16

There are two versions of AWQ: GEMM and GEMV. Both names relate to how matrix multiplication runs under the hood. We suggest the following:

- GEMV (quantized): 20% faster than GEMM, only batch size 1 (not good for large context).
- GEMM (quantized): Much faster than FP16 at batch sizes below 8 (good with large contexts).
- FP16 (non-quantized): Recommended for highest throughput: [vLLM](https://github.com/vllm-project/vllm).

#### Compute-bound vs Memory-bound

At small batch sizes with small 7B models, we are memory-bound. This means we are bound by the bandwidth our GPU has to push around the weights in memory, and this is essentially what limits how many tokens per second we can generate. Being memory-bound is what makes quantized models faster because your weights are 3x smaller and can therefore be pushed around in memory much faster. This is different from being compute-bound where the main time spent during generation is doing matrix multiplication. 

In the scenario of being compute-bound, which happens at higher batch sizes, you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation. This happens because AWQ quantized models only store the weights in INT4 but perform FP16 operations during inference, so we are essentially converting INT4 -> FP16 during inference.

### Fused modules

Fused modules are a large part of the speedup you get from AutoAWQ. The idea is to combine multiple layers into a single operation, thus becoming more efficient. Fused modules represent a set of custom modules that work separately from Huggingface models. They are compatible with `model.generate()` and other Huggingface methods, which comes with some inflexibility in how you can use your model if you activate fused modules:

- Fused modules are activated when you use `fuse_layers=True`.
- A custom cache is implemented. It preallocates based on batch size and sequence length.
    - You cannot change the sequence length after you have created your model.
    - Reference: `AutoAWQForCausalLM.from_quantized(max_new_tokens=seq_len, batch_size=batch_size)`
- The main accelerator in the fused modules comes from FasterTransformer, which is only compatible with Linux.
- The `past_key_values` from `model.generate()` are only dummy values, so they cannot be used after generation.

### Examples

More examples can be found in the [examples directory](examples).

<details>

<summary>Quantization</summary>

Expect this to take 10-15 minutes on smaller 7B models, and around 1 hour for 70B models.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

</details>

<details>

<summary>Inference</summary>

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/zephyr-7B-beta-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>"""

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

</details>

## Benchmarks

These benchmarks showcase the speed and memory usage of processing context (prefill) and generating tokens (decoding). The results include speed at various batch sizes and different versions of AWQ kernels. We have aimed to test models fairly using the same benchmarking tool that you can use to reproduce the results. Do note that speed may vary not only between GPUs but also between CPUs. What matters most is a GPU with high memory bandwidth and a CPU with high single core clock speed.

- Tested with AutoAWQ version 0.1.6
- GPU: RTX 4090 (AMD Ryzen 9 7950X)
- Command: `python examples/benchmark.py --model_path <hf_model> --batch_size 1`
- 🟢 for GEMV, 🔵 for GEMM, 🔴 for avoid using

| Model Name |  Size    | Version          | Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM)    |
|------------|----------|------------------|------------|----------------|---------------|------------------|-----------------|------------------|
| Vicuna     |   7B     | 🟢GEMV           | 1          | 64             | 64            | 639.65           | 198.848         | 4.50 GB (19.05%) |
| Vicuna     |   7B     | 🟢GEMV           | 1          | 2048           | 2048          | 1123.63          | 133.191         | 6.15 GB (26.02%) |
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| Mistral    |   7B     | 🔵GEMM           | 1          | 64             | 64            | 1093.35          | 156.317         | 4.35 GB (18.41%) |
| Mistral    |   7B     | 🔵GEMM           | 1          | 2048           | 2048          | 3897.02          | 114.355         | 5.55 GB (23.48%) |
| Mistral    |   7B     | 🔵GEMM           | 8          | 64             | 64            | 4199.18          | 1185.25         | 4.35 GB (18.41%) |
| Mistral    |   7B     | 🔵GEMM           | 8          | 2048           | 2048          | 3661.46          | 829.754         | 16.82 GB (71.12%)|
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| Mistral    |   7B     | 🟢GEMV           | 1          | 64             | 64            | 531.99           | 188.29          | 4.28 GB (18.08%) |
| Mistral    |   7B     | 🟢GEMV           | 1          | 2048           | 2048          | 903.83           | 130.66          | 5.55 GB (23.48%) |
| Mistral    |   7B     | 🔴GEMV           | 8          | 64             | 64            | 897.87           | 486.46          | 4.33 GB (18.31%) |
| Mistral    |   7B     | 🔴GEMV           | 8          | 2048           | 2048          | 884.22           | 411.893         | 16.82 GB (71.12%)|
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| TinyLlama  |   1B     | 🟢GEMV           | 1          | 64             | 64            | 1088.63          | 548.993         | 0.86 GB (3.62%)  |
| TinyLlama  |   1B     | 🟢GEMV           | 1          | 2048           | 2048          | 5178.98          | 431.468         | 2.10 GB (8.89%)  |
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| Llama 2    |   13B    | 🔵GEMM           | 1          | 64             | 64            | 820.34           | 96.74           | 8.47 GB (35.83%) |
| Llama 2    |   13B    | 🔵GEMM           | 1          | 2048           | 2048          | 2279.41          | 73.8213         | 10.28 GB (43.46%)|
| Llama 2    |   13B    | 🔵GEMM           | 3          | 64             | 64            | 1593.88          | 286.249         | 8.57 GB (36.24%) |
| Llama 2    |   13B    | 🔵GEMM           | 3          | 2048           | 2048          | 2226.7           | 189.573         | 16.90 GB (71.47%)|
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| MPT        |   7B     | 🔵GEMM           | 1          | 64             | 64            | 1079.06          | 161.344         | 3.67 GB (15.51%) |
| MPT        |   7B     | 🔵GEMM           | 1          | 2048           | 2048          | 4069.78          | 114.982         | 5.87 GB (24.82%) |
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| Falcon     |   7B     | 🔵GEMM           | 1          | 64             | 64            | 1139.93          | 133.585         | 4.47 GB (18.92%) |
| Falcon     |   7B     | 🔵GEMM           | 1          | 2048           | 2048          | 2850.97          | 115.73          | 6.83 GB (28.88%) |
| ...        |   ...    | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| CodeLlama  |   34B    | 🔵GEMM           | 1          | 64             | 64            | 681.74           | 41.01           | 19.05 GB (80.57%)|
| CodeLlama  |   34B    | 🔵GEMM           | 1          | 2048           | 2048          | 1072.36          | 35.8316         | 20.26 GB (85.68%)|
| ...        |  ...     | ...              | ...        | ...            | ...           | ...              | ...             | ...              |
| DeepSeek   |   33B    | 🔵GEMM           | 1          | 64             | 64            | 1160.18          | 40.29           | 18.92 GB (80.00%)|
| DeepSeek   |   33B    | 🔵GEMM           | 1          | 2048           | 2048          | 1012.1           | 34.0093         | 19.87 GB (84.02%)|

## Reference

If you find AWQ useful or relevant to your research, you can cite their [paper](https://arxiv.org/abs/2306.00978):

```
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
```
