import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
AUTOAWQ_VERSION = "0.1.8"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"

if not PYPI_BUILD:
    try:
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", torch.version.cuda).split("."))[:3]
        AUTOAWQ_VERSION += f"+cu{CUDA_VERSION}"
    except Exception as ex:
        raise RuntimeError("Your system must have an Nvidia GPU for installing AutoAWQ")

common_setup_kwargs = {
    "version": AUTOAWQ_VERSION,
    "name": "autoawq",
    "author": "Casper Hansen",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/casper-hansen/AutoAWQ",
    "keywords": ["awq", "autoawq", "quantization", "transformers"],
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}

requirements = [
    "torch>=2.0.1",
    "transformers>=4.35.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "sentencepiece",
    "lm_eval",
    "texttable",
    "toml",
    "attributedict",
    "protobuf",
    "torchvision",
    "tabulate"
]

def get_include_dirs():
    include_dirs = []

    conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)

    return include_dirs

def get_generator_flag():
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]
    
    return generator_flag

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cc = major * 10 + minor

        if cc < 75:
            raise RuntimeError("GPUs with compute capability less than 7.5 are not supported.")

    # figure out compute capability
    compute_capabilities = {75, 80, 86, 89, 90}

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

check_dependencies()
include_dirs = get_include_dirs()
generator_flags = get_generator_flag()
arch_flags = get_compute_capabilities()

if os.name == "nt":
    include_arch = os.getenv("INCLUDE_ARCH", "1") == "1"

    # Relaxed args on Windows
    if include_arch:
        extra_compile_args={"nvcc": arch_flags}
    else:
        extra_compile_args={}
else:
    extra_compile_args={
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
        "nvcc": [
            "-O3", 
            "-std=c++17",
            "-DENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ] + arch_flags + generator_flags
    }

extensions = [
    CUDAExtension(
        "awq_inference_engine",
        [
            "awq_cuda/pybind_awq.cpp",
            "awq_cuda/quantization/gemm_cuda_gen.cu",
            "awq_cuda/layernorm/layernorm.cu",
            "awq_cuda/position_embedding/pos_encoding_kernels.cu",
            "awq_cuda/quantization/gemv_cuda.cu"
        ], extra_compile_args=extra_compile_args
    )
]

if os.name != "nt":
    extensions.append(
        CUDAExtension(
            "ft_inference_engine",
            [
                "awq_cuda/pybind_ft.cpp",
                "awq_cuda/attention/ft_attention.cpp",
                "awq_cuda/attention/decoder_masked_multihead_attention.cu"
            ], extra_compile_args=extra_compile_args
        )
    )

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs
)
