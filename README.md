# AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |
</p>

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 2x while reducing memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.  AutoAWQ was created and improved upon from the [original work](https://github.com/mit-han-lab/llm-awq) from MIT.

*Latest News* 🔥
- [2023/09] Multi-GPU support, bug fixes, and better benchmark scripts available
- [2023/08] PyPi package released and AutoModel class available

## Install

Requirements: 
- Compute Capability 8.0 (sm80). Ampere and later architectures are supported.
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

Below, you will find examples of how to easily quantize a model and run inference.

### Quantization

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

### Inference

Run inference on a quantized model from Huggingface:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "casperhansen/vicuna-7b-v1.5-awq"
quant_file = "awq_model_w4_g128.pt"

model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

model.generate(...)
```

## Benchmarks

Benchmark speeds may vary from server to server and that it also depends on your CPU. If you want to minimize latency, you should rent a GPU/CPU combination that has high memory bandwidth for both and high single-core speed for CPU.

| Model       | GPU   | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----:|:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 4090  | 19.97             | 8.66              | 2.31x   |
| LLaMA-2-13B | 4090  | OOM               | 13.54             | --      |
| Vicuna-7B   | 4090  | 19.09             | 8.61              | 2.22x   |
| Vicuna-13B  | 4090  | OOM               | 12.17             | --      |
| MPT-7B      | 4090  | 17.09             | 12.58             | 1.36x   |
| MPT-30B     | 4090  | OOM               | 23.54             | --      |
| Falcon-7B   | 4090  | 29.91             | 19.84             | 1.51x   |
| LLaMA-2-7B  | A6000 | 27.14             | 12.44             | 2.18x   |
| LLaMA-2-13B | A6000 | 47.28             | 20.28             | 2.33x   |
| Vicuna-7B   | A6000 | 26.06             | 12.43             | 2.10x   |
| Vicuna-13B  | A6000 | 44.91             | 17.30             | 2.60x   |
| MPT-7B      | A6000 | 22.79             | 16.87             | 1.35x   |
| MPT-30B     | A6000 | OOM               | 31.57             | --      |
| Falcon-7B   | A6000 | 39.44             | 27.34             | 1.44x   |

<details>

<summary>Detailed benchmark (CPU vs. GPU)</summary>

Here is the difference between a fast and slow CPU on MPT-7B:

RTX 4090 + Intel i9 13900K (2 different VMs):
- CUDA 12.0, Driver 525.125.06: 134 tokens/s (7.46 ms/token)
- CUDA 12.0, Driver 525.125.06: 117 tokens/s (8.52 ms/token)

RTX 4090 + AMD EPYC 7-Series (3 different VMs):
- CUDA 12.2, Driver 535.54.03: 53 tokens/s (18.6 ms/token)
- CUDA 12.2, Driver 535.54.03: 56 tokens/s (17.71 ms/token)
- CUDA 12.0, Driver 525.125.06: 55 tokens/ (18.15 ms/token)

</details>

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
