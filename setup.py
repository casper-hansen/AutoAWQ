import os
from pathlib import Path
from torch.utils import cpp_extension
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.0.1",
    "name": "autoawq",
    "author": "Casper Hansen",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/casper-hansen/AutoAWQ",
    "keywords": ["awq", "autoawq", "quantization", "transformers"],
    "platforms": ["linux"],
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
    "torch>=2.0.0",
    "transformers>=4.32.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "sentencepiece",
    "lm_eval",
    "texttable",
    "toml",
    "attributedict",
    "protobuf",
    "torchvision"
]

include_dirs = []

conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
if os.path.isdir(conda_cuda_include_dir):
    include_dirs.append(conda_cuda_include_dir)

extensions = [
    cpp_extension.CppExtension(
        "awq_inference_engine",
        [
            "awq_cuda/pybind.cpp",
            "awq_cuda/quantization/gemm_cuda_gen.cu",
            "awq_cuda/layernorm/layernorm.cu",
            "awq_cuda/position_embedding/pos_encoding_kernels.cu"
        ], extra_compile_args={
            "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17"]
        }
    )
]

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': cpp_extension.BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs
)