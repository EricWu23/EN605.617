# Platform Overrides(Default this to be HOST, But we can overide it from command-line)
ARRAY_SIZE_X = 512
CFLAGS =-std=c++11 -DARRAY_SIZE_X[=$(ARRAY_SIZE_X)]
all:assignment.cu
	nvcc assignment.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o assignment.exe