ARCH ?= sm_35
LOWER_BOUND ?= 1
UPPER_BOUND ?= 16
all: test_transpose test_aos benchmark

test_transpose: test_transpose.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v test_transpose.cu -o test_transpose

test_aos: test_aos.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v test_aos.cu -o test_aos

benchmark: benchmark.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v benchmark.cu -o benchmark -DLOWER_BOUND=$(LOWER_BOUND) -DUPPER_BOUND=$(UPPER_BOUND)

reduce: reduce.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v reduce.cu -o reduce

construct: construct.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v construct.cu -o construct

transform: transform.cu
	nvcc -arch=$(ARCH) -I../ -Xptxas -v transform.cu -o transform
