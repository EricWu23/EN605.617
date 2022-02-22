VERBOSE = 1
CFLAGS =-std=c++11 -DVERBOSE[=$(VERBOSE)]
all: *.cu
	nvcc assignment.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o assignment.exe
	nvcc pageableHostmem_arithmetics_fake2D.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o pageableHostmem_arithmetics_fake2D
	nvcc pinnedHostmem_arithmetics_fake2D.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o pinnedHostmem_arithmetics_fake2D
.PHONY: clean
clean:
	rm -f assignment.exe pageableHostmem_arithmetics_fake2D pinnedHostmem_arithmetics_fake2D pinnedHostmem_arithmetics_2D global_memory host_memory