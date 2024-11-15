build:
	CUDA_VISIBLE_DEVICES=-1 pip install -e . -vvv
	
.PHONY: build