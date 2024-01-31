import os
import torch
import platform
import requests
import importlib.util
from pathlib import Path
from setuptools import setup, find_packages


def get_latest_kernels_version(repo):
    """
    Get the latest version of the kernels from the github repo.
    """
    response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest")
    data = response.json()
    tag_name = data["tag_name"]
    version = tag_name.replace("v", "")
    return version


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
    return f"https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v{release_version}/autoawq_kernels-{release_version}+{gpu_system_version}-cp{python_version}-cp{python_version}-{platform}_{architecture}.whl"


AUTOAWQ_VERSION = "0.1.8"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"

try:
    importlib.metadata.version("autoawq-kernels")
    KERNELS_INSTALLED = True
except importlib.metadata.PackageNotFoundError:
    KERNELS_INSTALLED = False

CUDA_VERSION = os.getenv("CUDA_VERSION", None) or torch.version.cuda
if CUDA_VERSION:
    CUDA_VERSION = "".join(CUDA_VERSION.split("."))[:3]

ROCM_VERSION = os.getenv("ROCM_VERSION", None) or torch.version.hip
if ROCM_VERSION:
    if ROCM_VERSION.startswith("5.6"):
        ROCM_VERSION = "5.6.1"
    elif ROCM_VERSION.startswith("5.7"):
        ROCM_VERSION = "5.7.1"

    ROCM_VERSION = "".join(ROCM_VERSION.split("."))[:3]


if not PYPI_BUILD:
    if CUDA_VERSION:
        AUTOAWQ_VERSION += f"+cu{CUDA_VERSION}"
    elif ROCM_VERSION:
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

# kernels can be downloaded from pypi for cuda+121+(linux or windows) only
# for everything else, we need to download the kernels wheel from github
if platform.system().lower() != "darwin" and not KERNELS_INSTALLED:
    if CUDA_VERSION.startswith("12"):
        requirements.append("autoawq-kernels")
    elif CUDA_VERSION.startswith("11") or ROCM_VERSION in ["561", "571"]:
        gpu_system_version = (
            f"cu{CUDA_VERSION}" if CUDA_VERSION else f"rocm{ROCM_VERSION}"
        )
        kernels_version = get_latest_kernels_version("casper-hansen/AutoAWQ_kernels")
        python_version = "".join(platform.python_version_tuple()[:2])
        platform_name = platform.system().lower()
        architecture = platform.machine().lower()
        latest_rocm_kernels_wheels = get_kernels_whl_url(
            gpu_system_version,
            kernels_version,
            python_version,
            platform_name,
            architecture,
        )
        requirements.append(f"autoawq-kernels@{latest_rocm_kernels_wheels}")
    else:
        raise RuntimeError(
            "Your system must have either Nvidia or AMD GPU to build this package."
        )

setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "eval": ["lm_eval>=0.4.0", "tabulate", "protobuf", "evaluate", "scipy"],
    },
    **common_setup_kwargs,
)
