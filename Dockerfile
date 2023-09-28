# -------------------------------------------------------------------------------------
# PYTHON BASE IMAGE. 
# Runs: Installs all dependencies required for running the reader. 
# Entrypoint: runs the run.sh script from the reader. 
# -------------------------------------------------------------------------------------
# base image with cuda 11.7 (need this specifically for awq)
FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04 as python-base

# install python (base image doesn't have this)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    wget \
    git \
    socat \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root


WORKDIR /code

RUN mkdir /code/models

RUN pip install torch==2.0.1


# install awq lib and kernels
RUN pip install setuptools --upgrade
RUN pip3 install --upgrade pip

# install awq from github 
# RUN git clone https://github.com/casper-hansen/AutoAWQ
# RUN cd AutoAWQ && git checkout 7fbe9bbc98fc48a5b6247bd9dddf7366b82d33f3 && pip install -e .

COPY . /code/AutoAWQ

# run using the gemv kernels
ARG USE_GEMV=0

RUN cd /code/AutoAWQ && USE_GEMV=${USE_GEMV} pip install -e .

ENTRYPOINT [ "/bin/bash" ]