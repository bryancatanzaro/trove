ARCH ?= sm_70
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16
TILE_SIZE ?= 32

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

all: benchmark block

benchmark: benchmark.cu
	$(NVCC) -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND) -DTILE_SIZE=$(TILE_SIZE)

block: block.cu
	$(NVCC) -arch=$(ARCH) -I../ -Xptxas -v block.cu -o block -DTILE_SIZE=$(TILE_SIZE)

clean:
	rm -f *.o benchmark block
