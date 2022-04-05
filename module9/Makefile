include sources.mk


VERBOSE = 0
MAXOPERIONS =1000
CFLAGS =-std=c++14 --device-c -DVERBOSE[=$(VERBOSE)] -DMAXOPERIONS[=$(MAXOPERIONS)] $(INCLUDES:%=-I %)

CC = nvcc
LD= nvcc

OBJS = $(SOURCES:.cu=.o)
all: *.cu
	$(CC) $(CFLAGS) $(SOURCESLOCATION)
	$(CC) $(OBJS) -L /usr/local/cuda/lib -lcudart -o assignment
.PHONY: clean
clean:
	rm -f $(OBJS) assignment $(IRRELEVANT)