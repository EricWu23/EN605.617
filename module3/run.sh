#!/bin/bash

echo "My first shell script"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make
# notice that the branchingimpact1 is built without passing -DARRAY_SIZE_X[=512]. This will build it with the maro ARRAY_SIZE_X defined as 32
# for the purpose of investigating the warp divergence due to branching

echo "----------------------------------------------------------------------------"
echo "Test 1:Parallel Computing- Array mathematics, Add, Subtract, Multiply, Mod:"
echo "----------------------------------------------------------------------------"

assignment.exe 512 256

echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 1.1.1: Parallel Computing- Array mathematics with different array size, num of threads :"
echo "-----------------------------------------------------------------------------------------------------------"

# notice that to execute properly, the total number of thread has to match the underline array size. So a recompile with a redefined  ARRAY_SIZE_X is needed.
nvcc assignment.cu -std=c++11 -DARRAY_SIZE_X[=256] -L /usr/local/cuda/lib -lcudart -o assignment.exe

assignment.exe 256 256

echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 1.1.2: Parallel Computing- Array mathematics with different array size, num of threads :"
echo "-----------------------------------------------------------------------------------------------------------"
# notice that to execute properly, the total number of thread has to match the underline array size. So a recompile with a redefined  ARRAY_SIZE_X is needed.
nvcc assignment.cu -std=c++11 -DARRAY_SIZE_X[=1024] -L /usr/local/cuda/lib -lcudart -o assignment.exe

assignment.exe 1024 256


echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 1.2.1: Parallel Computing- Array mathematics with a different blocksize :"
echo "-----------------------------------------------------------------------------------------------------------"

nvcc assignment.cu -std=c++11 -DARRAY_SIZE_X[=512] -L /usr/local/cuda/lib -lcudart -o assignment.exe

assignment.exe 512 512

echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 1.2.2: Parallel Computing- Array mathematics with a different blocksize :"
echo "-----------------------------------------------------------------------------------------------------------"

# notice that to solely changing the blocksize does not require the recompile.

assignment.exe 512 128

echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 2.1: Branching impact without warp divergence :"
echo "-----------------------------------------------------------------------------------------------------------"

branchingimpact.exe 512 256

# compare with the assignment.exe 512 256

echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 2.2: Branching impact with warp divergence :"
echo "-----------------------------------------------------------------------------------------------------------"
branchingimpact1.exe 32 32