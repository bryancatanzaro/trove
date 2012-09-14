ARCH ?= sm_35

test: test.cu oet.h bubble.h
	nvcc -arch=$(ARCH) -Xptxas -v test.cu -o test