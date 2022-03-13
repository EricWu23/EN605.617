include sources.mk


VERBOSE = 0
MAXOPERIONS =1000
CFLAGS =-std=c++11 --device-c -DVERBOSE[=$(VERBOSE)] -DMAXOPERIONS[=$(MAXOPERIONS)]

CC = nvcc
LD= nvcc

OBJS = $(SOURCES:.cu=.o)
all: *.cu
	$(CC) $(CFLAGS) $(SOURCES)
	$(CC) $(OBJS) -L /usr/local/cuda/lib -lcudart -o assignment
.PHONY: clean
clean:
	rm -f $(OBJS) assignment $(IRRELEVANT)