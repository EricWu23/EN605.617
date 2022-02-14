# Platform Overrides(Default this to be HOST, But we can overide it from command-line)
ARRAY_SIZE_X = 512
CFLAGS =-std=c++11 -DARRAY_SIZE_X[=$(ARRAY_SIZE_X)]
all:assignment.cu
	nvcc assignment.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o assignment.exe
	nvcc branchingimpact.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o branchingimpact.exe
	nvcc branchingimpact1.cu -std=c++11 -L /usr/local/cuda/lib -lcudart -o branchingimpact1.exe
.PHONY: clean
clean:
	rm -f assignment.exe branchingimpact.exe branchingimpact1.exe