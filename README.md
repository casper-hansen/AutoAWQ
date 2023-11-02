# AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |

</p>
<p align="center">
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_600+_models_available-8A2BE2">
    </a>
    <a href="https://github.com/casper-hansen/AutoAWQ/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/release/casper-hansen/AutoAWQ.svg">
    </a>
    <a href="https://pypi.org/project/autoawq/">
        <img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/autoawq/month">
    </a>
</p>

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 2x while reducing memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.  AutoAWQ was created and improved upon from the [original work](https://github.com/mit-han-lab/llm-awq) from MIT.

*Latest News* 🔥
- [2023/11] AutoAWQ has been merged into 🤗 transformers. Example found in: [examples/basic_transformers](examples/basic_transformers.py).
- [2023/10] Mistral (Fused Modules), Bigcode, Turing support, Memory Bug Fix (Saves 2GB VRAM)
- [2023/09] 1.6x-2.5x speed boost on fused models (now including MPT and Falcon).
- [2023/09] Multi-GPU support, bug fixes, and better benchmark scripts available
- [2023/08] PyPi package released and AutoModel class available

## Install

Requirements: 
- Compute Capability 7.5 (sm75). Turing and later architectures are supported.
- CUDA Toolkit 11.8 and later.

---

Install:
- Use pip to install awq

```
pip install autoawq
```

### Using conda

CUDA dependencies can be hard to manage sometimes. It is recommended to use conda with AutoAWQ:

```
conda create --name autoawq python=3.10 -y
conda activate autoawq
conda install pytorch=2.0.1 torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install autoawq
```

### Build source

<details>

<summary>Build AutoAWQ from scratch</summary>

Build time can take 10 minutes. Download your model while you install AutoAWQ.

```
git clone https://github.com/casper-hansen/AutoAWQ
cd AutoAWQ
pip install -e .
```

</details>

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

## Usage

Under examples, you can find examples of how to quantize, run inference, and benchmark AutoAWQ models.

### INT4 GEMM vs INT4 GEMV vs FP16

There are two versions of AWQ: GEMM and GEMV. Both names relate to how matrix multiplication runs under the hood. We suggest the following:

- GEMV (quantized): 20% faster than GEMM for small batch sizes (max batch size 4 / small context).
- GEMM (quantized): Much faster than FP16 at batch sizes below 8 (good with large contexts).
- FP16 (non-quantized): Recommended for highest throughput: [vLLM](https://github.com/vllm-project/vllm).

#### Compute-bound vs Memory-bound

At small batch sizes with small 7B models, we are memory-bound. This means we are bound by the bandwidth our GPU has to push around the weights in memory, and this is essentially what limits how many tokens per second we can generate. Being memory-bound is what makes quantized models faster because your weights are 3x smaller and can therefore be pushed around in memory much faster. This is different from being compute-bound where the main time spent during generation is doing matrix multiplication. 

In the scenario of being compute-bound, which happens at higher batch sizes, you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation. This happens because AWQ quantized models only store the weights in INT4 but perform FP16 operations during inference, so we are essentially converting INT4 -> FP16 during inference.

### Fused modules

Fused modules are a large part of the speedup you get from AutoAWQ. The idea is to combine multiple layers into a single operation, thus becoming more efficient. Fused modules represent a set of custom modules that work separately from Huggingface models. They are compatible with `model.generate()` and other Huggingface methods, which comes with some inflexibility in how you can use your model if you activate fused modules:

- Fused modules are activated when you use `fuse_layers=True`.
- A custom cache is implemented. It preallocates based on batch size and sequence length.
    - You cannot change the sequence length or batch size after you have created your model.
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

quant_path = "casperhansen/vicuna-7b-v1.5-awq"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:"""

tokens = tokenizer(
    prompt_template.format(prompt="How are you today?"), 
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

<details>

<summary>AutoAWQForCausalLM.from_quantized</summary>

- `quant_path`: Path to folder containing model files.
- `quant_filename`: The filename to model weights or `index.json` file.
- `max_new_tokens`: The max sequence length, used to allocate kv-cache for fused models.
- `fuse_layers`: Whether or not to use fused layers.
- `batch_size`: The batch size to initialize the AWQ model with.

</details>

## Benchmarks

| Model Name    | Version | Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM)    |
|---------------|---------|------------|----------------|---------------|------------------|-----------------|------------------|
| Vicuna 7B     | GEMM    | 1          | 64             | 64            | 2618.88          | 125.428         | 4.57 GB (19.31%) |
| Vicuna 7B     | GEMM    | 1          | 128            | 128           | 2808.09          | 123.865         | 4.61 GB (19.44%) |
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |
| Vicuna 7B     | GEMV    | 1          | 64             | 64            | 233.909          | 154.475         | 4.66 GB (19.68%) |
| Vicuna 7B     | GEMV    | 1          | 128            | 128           | 233.145          | 152.133         | 4.66 GB (19.68%) |
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |
| MPT 7B        | GEMM    | 1          | 64             | 64            | 2752.9           | 120.772         | 3.67 GB (15.48%) |
| MPT 7B        | GEMM    | 1          | 128            | 128           | 2982.67          | 119.52          | 3.70 GB (15.61%) |
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |
| MPT 7B        | GEMV    | 1          | 64             | 64            | 241.026          | 136.476         | 3.67 GB (15.48%) |
| MPT 7B        | GEMV    | 1          | 128            | 128           | 239.44           | 137.599         | 3.70 GB (15.61%) |
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |
| Falcon 7B     | GEMM    | 1          | 64             | 64            | 1920.61          | 94.5963         | 4.48 GB (18.92%) |
| Falcon 7B     | GEMM    | 1          | 128            | 128           | 2406.1           | 94.793          | 4.48 GB (18.92%) |
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |
| Aquila2 34B   | GEMM    | 1          | 64             | 64            | 516.544          | 23.3536         | 18.26 GB (46.12%)|
| Aquila2 34B   | GEMM    | 1          | 128            | 128           | 643.968          | 23.3803         | 18.26 GB (46.12%)|
| ...           | ...     | ...        | ...            | ...           | ...              | ...             | ...              |

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
