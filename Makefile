ARCH ?= sm_35
all: test_transpose test_aos benchmark

test_transpose: test_transpose.cu transpose.h
	nvcc -arch=$(ARCH) -Xptxas -v test_transpose.cu -o test_transpose --maxrregcount=32

test_aos: test_aos.cu
	nvcc -arch=$(ARCH) -Xptxas -v test_aos.cu -o test_aos --maxrregcount=32

benchmark: benchmark.cu
	nvcc -arch=$(ARCH) -Xptxas -v benchmark.cu -o benchmark --maxrregcount=32