# AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |

</p>
<p align="center">
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_400+_models_available-8A2BE2">
    </a>
    <a href="https://github.com/casper-hansen/AutoAWQ/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/release/casper-hansen/AutoAWQ.svg">
    </a>
    <a href="https://pypi.org/project/autoawq/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/autoawq">
    </a>
</p>

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 2x while reducing memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.  AutoAWQ was created and improved upon from the [original work](https://github.com/mit-han-lab/llm-awq) from MIT.

*Latest News* 🔥
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
| Vicuna   | 7B/13B                      |
| MPT      | 7B/30B                      |
| Falcon   | 7B/40B                      |
| OPT      | 125m/1.3B/2.7B/6.7B/13B/30B |
| Bloom    | 560m/3B/7B/                 |
| GPTJ     | 6.7B                        |

## Usage

Under examples, you can find examples of how to quantize, run inference, and benchmark AutoAWQ models.

### INT4 GEMM vs INT4 GEMV vs FP16

There are two versions of AWQ: GEMM and GEMV. Both names relate to how matrix multiplication runs under the hood. We suggest the following:

- GEMV (quantized): Best for small context, batch size 1, highest number of tokens/s.
- GEMM (quantized): Best for larger context, up to batch size 8, faster than GEMV on batch size > 1, slower than GEMV on batch size = 1.
- FP16 (non-quantized): Best for large batch sizes of 8 or larger, highest throughput. We recommend [TGI](https://github.com/huggingface/text-generation-inference) or [vLLM](https://github.com/vllm-project/vllm).

### Examples

<details>

<summary>Quantization</summary>

Expect this to take 10-15 minutes on smaller 7B models, and around 1 hour for 70B models.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }

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
quant_file = "awq_model_w4_g128.pt"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file, fuse_layers=True)
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

### Vicuna 7B (LLaMa-2)

- Note: Blazing fast generation, slow context processing
- GPU: NVIDIA GeForce RTX 3090
- Version: GEMV
- Command: `python examples/benchmark.py --model_path casperhansen/vicuna-7b-v1.5-awq-gemv`

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)    |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:-----------------|
|            1 |               32 |              32 |           231.393  |           153.632 | 4.66 GB (19.68%) |
|            1 |               64 |              64 |           233.909  |           154.475 | 4.66 GB (19.68%) |
|            1 |              128 |             128 |           233.145  |           152.133 | 4.66 GB (19.68%) |
|            1 |              256 |             256 |           228.562  |           147.692 | 4.67 GB (19.72%) |
|            1 |              512 |             512 |           228.914  |           139.179 | 4.80 GB (20.26%) |
|            1 |             1024 |            1024 |           227.393  |           125.058 | 5.56 GB (23.48%) |
|            1 |             2048 |            2048 |           225.736  |           123.228 | 8.08 GB (34.09%) |

- Note: Fast generation, fast context processing
- GPU: NVIDIA GeForce RTX 3090
- Version: GEMM
- Command: `python examples/benchmark.py --model_path casperhansen/vicuna-7b-v1.5-awq`

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)    |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:-----------------|
|            1 |               32 |              32 |            521.444 |           126.51  | 4.55 GB (19.21%) |
|            1 |               64 |              64 |           2618.88  |           125.428 | 4.57 GB (19.31%) |
|            1 |              128 |             128 |           2808.09  |           123.865 | 4.61 GB (19.44%) |
|            1 |              256 |             256 |           2807.46  |           120.779 | 4.67 GB (19.72%) |
|            1 |              512 |             512 |           2769.9   |           115.08  | 4.80 GB (20.26%) |
|            1 |             1024 |            1024 |           2640.95  |           105.493 | 5.56 GB (23.48%) |
|            1 |             2048 |            2048 |           2341.36  |           104.188 | 8.08 GB (34.09%) |

### MPT 7B

- Note: Blazing fast generation, slow context processing
- GPU: NVIDIA GeForce RTX 3090
- Command: `python examples/benchmark.py --model_path casperhansen/mpt-7b-8k-chat-awq-gemv`
- Version: GEMV

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)    |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:-----------------|
|            1 |               32 |              32 |            187.332 |           136.765 | 3.65 GB (15.42%) |
|            1 |               64 |              64 |            241.026 |           136.476 | 3.67 GB (15.48%) |
|            1 |              128 |             128 |            239.44  |           137.599 | 3.70 GB (15.61%) |
|            1 |              256 |             256 |            233.184 |           137.02  | 3.76 GB (15.88%) |
|            1 |              512 |             512 |            233.082 |           135.633 | 3.89 GB (16.41%) |
|            1 |             1024 |            1024 |            231.504 |           122.197 | 4.40 GB (18.57%) |
|            1 |             2048 |            2048 |            228.307 |           121.468 | 5.92 GB (24.98%) |

- Note: Fast generation, fast context processing
- GPU: NVIDIA GeForce RTX 3090
- Version: GEMM
- Command: `python examples/benchmark.py --model_path casperhansen/mpt-7b-8k-chat-awq`

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)    |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:-----------------|
|            1 |               32 |              32 |            557.714 |           118.567 | 3.65 GB (15.42%) |
|            1 |               64 |              64 |           2752.9   |           120.772 | 3.67 GB (15.48%) |
|            1 |              128 |             128 |           2982.67  |           119.52  | 3.70 GB (15.61%) |
|            1 |              256 |             256 |           3009.16  |           116.911 | 3.76 GB (15.88%) |
|            1 |              512 |             512 |           2901.91  |           111.607 | 3.95 GB (16.68%) |
|            1 |             1024 |            1024 |           2718.68  |           102.623 | 4.40 GB (18.57%) |
|            1 |             2048 |            2048 |           2363.61  |           101.368 | 5.92 GB (24.98%) |

### Falcon 7B

- Note: Fast generation, fast context processing
- GPU: NVIDIA GeForce RTX 3090
- Command: `python examples/benchmark.py --model_path casperhansen/falcon-7b-awq --quant_file awq_model_w4_g64.pt`
- Version: GEMM

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)    |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:-----------------|
|            1 |               32 |              32 |            466.826 |           95.1413 | 4.47 GB (18.88%) |
|            1 |               64 |              64 |           1920.61  |           94.5963 | 4.48 GB (18.92%) |
|            1 |              128 |             128 |           2406.1   |           94.793  | 4.48 GB (18.92%) |
|            1 |              256 |             256 |           2521.08  |           94.1144 | 4.48 GB (18.92%) |
|            1 |              512 |             512 |           2478.28  |           93.4123 | 4.48 GB (18.92%) |
|            1 |             1024 |            1024 |           2256.22  |           94.0237 | 4.69 GB (19.78%) |
|            1 |             2048 |            2048 |           1831.71  |           94.2032 | 6.83 GB (28.83%) |

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
