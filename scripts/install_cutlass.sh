#!/bin/bash

# Download CUTLASS v2.11.0 (Not compatible with 3.x)
git clone --depth 1 --branch v2.11.0 https://github.com/NVIDIA/cutlass.git
cd cutlass
rm -rf build
mkdir -p build && cd build

# Install CUTLASS
export CUDACXX=/usr/local/cuda/bin/nvcc

# NOTE: You can build for all Ampere+Hopper with 80;86;89;90
cmake .. -DCUTLASS_NVCC_ARCHS="80" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j $(nproc)

# Copy over the files
cp -r ../include/cutlass /usr/local/include/
mkdir -p /usr/local/include/cutlass/util
cp -r ../tools/util/include/cutlass/util/* /usr/local/include/cutlass/util/
ldconfig
rm -rf cutlass