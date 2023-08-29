import os
import torch
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    compute_capabilities = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("GPUs with compute capability less than 8.0 are not supported.")
        compute_capabilities.add(major * 10 + minor)
    
    # figure out compute capability
    compute_capabilities = {80, 86, 89, 90}
    
    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]
    
    return capability_flags

# Define dependencies
dependencies = [
    "accelerate", "sentencepiece", "tokenizers>=0.12.1",
    "transformers>=4.32.0", 
    "lm_eval", "texttable",
    "toml", "attributedict",
    "protobuf",
    "torch>=2.0.0", "torchvision"
]

# Get environment variables
build_cuda_extension = os.environ.get('BUILD_CUDA_EXT', '1') == '1'

# Setup CUDA extension
ext_modules = []

if build_cuda_extension:
    # num threads
    n_threads = str(min(os.cpu_count(), 8))

    # final args
    capability_flags = get_compute_capabilities()
    cxx_args = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"]
    nvcc_args = ["-O3", "-std=c++17", "--threads", n_threads] + capability_flags

    ext_modules.append(
        CUDAExtension(
            name="awq_inference_engine",
            sources=[
                "awq_cuda/pybind.cpp",
                "awq_cuda/quantization/gemm_cuda_gen.cu",
                "awq_cuda/layernorm/layernorm.cu",
                "awq_cuda/position_embedding/pos_encoding_kernels.cu"
            ],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args
            },
        )
    )

# Find directories to be included in setup
include_dirs = []
conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

if os.path.isdir(conda_cuda_include_dir):
    include_dirs.append(conda_cuda_include_dir)

setup(
    name="autoawq",
    version="0.1.0",
    author="Casper Hansen",
    license="MIT",
    description="AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    url="https://github.com/casper-hansen/AutoAWQ",
    keywords=["awq", "autoawq", "quantization", "transformers"],
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    install_requires=dependencies,
    include_dirs=include_dirs,
    packages=find_packages(exclude=["examples*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
