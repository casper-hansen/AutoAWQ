git clone --recurse-submodules https://github.com/casper-hansen/torch-int.git
cd torch-int
sh build_cutlass.sh
pip install -v -e .