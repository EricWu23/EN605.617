include sources.mk

CFLAGS =-std=c++11 $(INCLUDES:%=-I %)

CC = g++
LD= g++

OBJS = $(SOURCES:.cpp=.o)
all: *.cpp
	$(CC) $(CFLAGS) $(SOURCESLOCATION) -c
	$(LD) $(OBJS) -L /usr/local/cuda/lib64/ -lOpenCL -o assignment
.PHONY: clean
clean:
	rm -f $(OBJS) assignment $(IRRELEVANT)