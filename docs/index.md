# AutoAWQ

AutoAWQ pushes ease of use and fast inference speed into one package. In the following documentation,
you will learn how to quantize and run inference.

Example inference speed (RTX 4090, Ryzen 9 7950X, 64 tokens):

- Vicuna 7B (GEMV kernel): 198.848 tokens/s
- Mistral 7B (GEMM kernel): 156.317 tokens/s
- Mistral 7B (ExLlamaV2 kernel): 188.865 tokens/s
- Mixtral 46.7B (GEMM kernel): 93 tokens/s (2x 4090)

## Installation notes

- Install: `pip install autoawq`.
- Your torch version must match the build version, i.e. you cannot use torch 2.0.1 with a wheel that was built with 2.2.0.
- For AMD GPUs, inference will run through ExLlamaV2 kernels without fused layers. You need to pass the following arguments to run with AMD GPUs:

```python
model = AutoAWQForCausalLM.from_quantized(
    ...,
    fuse_layers=False,
    use_exllama_v2=True
)
```

## Supported models

The detailed support list:

| Models   | Sizes                       |
| -------- | --------------------------- |
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