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
- For CPU device, you should install intel-extension-for-transformers with `pip install intel-extension-for-transformers`. And the latest version of torch is required since "intel-extension-for-transformers(ITREX)" was built with the latest version of torch(now ITREX 1.4 was build with torch 2.2). If you build ITREX from source code, then you need to ensure the consistency of the torch version. And you should use "use_qbits=True" for CPU device. Up to now, the feature of fuse_layers hasn't been ready for CPU device.
    ```python
    model = AutoAWQForCausalLM.from_quantized(
        ...,
        fuse_layers=False,
        use_qbits=True
    )
    ```

## Supported models

We support modern LLMs. You can find a list of supported Huggingface `model_types` in `awq/models`.