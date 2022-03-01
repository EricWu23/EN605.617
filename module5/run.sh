#!/bin/bash

echo "Module5: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make
# notice that the default operation for make will build a Non-verbose option, which only output execution speed timming info. 
# If you want to see the actual arrays and the result of computation, you should use make VERBOSE=1
# another thing can be changed using make is the MAXOPERIONS, for example
# make MAXOPERIONS=10  will compile a version that will run the same operation (for example, array add) 10 times inside a kernel.
# Increase this can increase the portion of computation in the total amount of time spend by the kernel. 
# for shared memory and constant memory, there might be overhead. Set MAXOPERIONS=1 to see the time associated with the overhead.

echo "----------------------------------------------------------------------------"
echo "Test 1: four previous math opertions using global memory, shared memory, constant memory"
echo "----------------------------------------------------------------------------"

echo "----------------------------------------------------------------------------"
echo "Test 1.1: perform math operation on two arrays with 1048576 elements 1000 times . blocksize=256 "
echo "----------------------------------------------------------------------------"
./assignment

echo "----------------------------------------------------------------------------"
echo "Test 1.2: perform math operation on two arrays with 1048576 elements 1 times . blocksize=256 "
echo "----------------------------------------------------------------------------"
make MAXOPERIONS=1
./assignment


echo "----------------------------------------------------------------------------"
echo "Test 1.3: perform math operation on two arrays with 1048576 elements 1000 times . blocksize=1024 "
echo "----------------------------------------------------------------------------"
make
./assignment 1048576 1024

echo "----------------------------------------------------------------------------"
echo "Test 1.3: perform math operation on two arrays with 1048576 elements 1000 times . blocksize=32 "
echo "----------------------------------------------------------------------------"
./assignment 1048576 32

echo "----------------------------------------------------------------------------"
echo "Test 1.4: perform math operation on two arrays with 2048 elements 1000 times . blocksize=256 "
echo "----------------------------------------------------------------------------"
./assignment 2048 256

echo "----------------------------------------------------------------------------"
echo "Test 1.5: perform math operation on two arrays with 2048 elements 10000 times . blocksize=256 "
echo "----------------------------------------------------------------------------"
make MAXOPERIONS=10000
./assignment 2048 256