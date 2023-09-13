# AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |
</p>

AutoAWQ is an easy-to-use package for 4-bit quantized models. AutoAWQ speeds up models by 2x while reducing memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs.  AutoAWQ was created and improved upon from the [original work](https://github.com/mit-han-lab/llm-awq) from MIT.

*Latest News* 🔥
- [2023/09] 1.6x-2.5x speed boost on fused models (now including MPT and Falcon). LLaMa 7B - up to 180 tokens/s.
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

<details>

<summary>Quantization</summary>

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
