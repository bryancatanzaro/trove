ARCH ?= sm_35

test: test.cu transpose.h
	nvcc -arch=$(ARCH) -Xptxas -v test.cu -o test --maxrregcount=32