include sources.mk


CFLAGS =-std=c++14 --device-c  $(INCLUDES:%=-I %)

CC = nvcc
LD= nvcc

OBJS = $(SOURCES:.cu=.o)
all: *.cu
	$(CC) $(CFLAGS) $(SOURCESLOCATION)
	$(LD) $(OBJS) $(LIBRARIESNVGRAPH) -lcudart -o assignment
.PHONY: clean
clean:
	rm -f $(OBJS) assignment $(IRRELEVANT)