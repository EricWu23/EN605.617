include sources.mk


HA = 2
WA = 9
WB = 2
HA2 = 3
WA2 = 3
NRHS = 1
CFLAGS =-std=c++11 --device-c -DHA[=$(HA)] -DWA[=$(WA)] -DWB[=$(WB)] -DHA2[=$(HA2)] -DWA2[=$(WA2)] -DNRHS[=$(NRHS)] $(INCLUDES:%=-I %)

CC = nvcc
LD= nvcc

OBJS = $(SOURCES:.cu=.o)
all: *.cu
	$(CC) $(CFLAGS) $(SOURCESLOCATION)
	$(LD) $(OBJS) -L /usr/local/cuda/lib64 -lcublas -lcudart -lcusolver -o assignment
.PHONY: clean
clean:
	rm -f $(OBJS) assignment $(IRRELEVANT)