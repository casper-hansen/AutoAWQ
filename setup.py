import os
import torch
from pathlib import Path
from setuptools import setup, find_packages

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
AUTOAWQ_VERSION = "0.1.8"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"
CUDA_VERSION = os.getenv("CUDA_VERSION", None) or torch.version.cuda
ROCM_VERSION = os.environ.get("ROCM_VERSION", None) or torch.version.hip


if not PYPI_BUILD:
    # only adding CUDA/ROCM version if we are not building for PyPI to comply with PEP 440
    if CUDA_VERSION:
        CUDA_VERSION = "".join(CUDA_VERSION.split("."))[:3]
        AUTOAWQ_VERSION += f"+cu{CUDA_VERSION}"
    elif ROCM_VERSION:
        ROCM_VERSION = "".join(ROCM_VERSION.split("."))[:3]
        AUTOAWQ_VERSION += f"+rocm{ROCM_VERSION}"
    else:
        raise RuntimeError(
            "Your system must have either Nvidia or AMD GPU to build this package."
        )

common_setup_kwargs = {
    "version": AUTOAWQ_VERSION,
    "name": "autoawq",
    "author": "Casper Hansen",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference.",
    "long_description": (Path(__file__).parent / "README.md").read_text(
        encoding="UTF-8"
    ),
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
    ],
}

requirements = [
    "autoawq-kernels",
    "torch>=2.0.1",
    "transformers>=4.35.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "datasets",
]

setup(packages=find_packages(), install_requires=requirements, **common_setup_kwargs)
