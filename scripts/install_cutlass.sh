export CUDACXX=/usr/local/cuda/bin/nvcc

cd submodules/cutlass
rm -rf build
mkdir -p build && cd build

# NOTE: You can build for all Ampere+Hopper with 80;86;89;90
cmake .. -DCUTLASS_NVCC_ARCHS="80" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j $(nproc)