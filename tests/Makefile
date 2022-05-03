ARCH ?= sm_70
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

all: benchmark block

benchmark: benchmark.cu
	$(NVCC) -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)

benchmark_3d: benchmark_3d.cu
	$(NVCC) -arch=$(ARCH) -I../ -Xptxas -v benchmark_3d.cu -o benchmark_3d -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)

block: block.cu
	$(NVCC) -arch=$(ARCH) -I../ -Xptxas -v block.cu -o block
