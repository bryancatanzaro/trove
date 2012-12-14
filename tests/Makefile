ARCH ?= sm_35
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16

all: benchmark construct transform copy

benchmark: benchmark.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)

construct: construct.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v construct.cu -o construct

transform: transform.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v transform.cu -o transform

copy: copy.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v copy.cu -o copy

block: block.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v block.cu -o block
