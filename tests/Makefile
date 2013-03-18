ARCH ?= sm_35
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16

all: benchmark block

benchmark: benchmark.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)

block: block.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v block.cu -o block 
