VERBOSE = 0
MAXOPERIONS =1000
CFLAGS =-std=c++11 -DVERBOSE[=$(VERBOSE)] -DMAXOPERIONS[=$(MAXOPERIONS)]
all: *.cu
	nvcc assignment.cu $(CFLAGS) -L /usr/local/cuda/lib -lcudart -o assignment
.PHONY: clean
clean:
	rm -f assignment register module6_stretch_problem