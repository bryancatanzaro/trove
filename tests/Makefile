ARCH ?= sm_35
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16

all: benchmark

benchmark: benchmark.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)
