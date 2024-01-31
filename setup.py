import os
import torch
import platform
import requests
from pathlib import Path
from setuptools import setup, find_packages


def get_latest_kernels_version(repo):
    """
    Get the latest version of the kernels from the github repo.
    """
    response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest")
    data = response.json()
    return data["tag_name"]


def get_kernels_whl_url(
    gpu_system_version,
    release_version,
    python_version,
    platform,
    architecture,
):
    """
    Get the url for the kernels wheel file.
    """
    return f"https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/{release_version}/autoawq_kernels-{release_version[1:]}+{gpu_system_version}-{python_version}-{python_version}-{platform}_{architecture}.whl"


AUTOAWQ_VERSION = "0.1.8"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"
CUDA_VERSION = os.getenv("CUDA_VERSION", None) or torch.version.cuda
ROCM_VERSION = os.getenv("ROCM_VERSION", None) or torch.version.hip

if isinstance(ROCM_VERSION, str):
    if ROCM_VERSION.startswith("5.6"):
        ROCM_VERSION = "5.6.1"
    elif ROCM_VERSION.startswith("5.7"):
        ROCM_VERSION = "5.7.1"


if not PYPI_BUILD:
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
    "torch>=2.0.1",
    "transformers>=4.35.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "datasets",
]


# kernels can be downloaded from pypi for cuda+(linux or windows)
if platform.system().lower() != "darwin" and CUDA_VERSION:
    requirements.append("autoawq-kernels")
elif platform.system().lower() != "darwin" and ROCM_VERSION:
    kernels_version = get_latest_kernels_version("casper-hansen/AutoAWQ_kernels")
    python_version = ".".join(platform.python_version_tuple()[:2])
    platform_name = platform.system().lower()
    architecture = platform.machine().lower()
    latest_rocm_kernels_wheels = get_kernels_whl_url(
        f"rocm{ROCM_VERSION}",
        kernels_version,
        python_version,
        platform_name,
        architecture,
    )
    requirements.append(latest_rocm_kernels_wheels)

setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "eval": ["lm_eval>=0.4.0", "tabulate", "protobuf", "evaluate", "scipy"],
    },
    **common_setup_kwargs,
)
