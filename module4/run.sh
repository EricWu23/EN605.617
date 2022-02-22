#!/bin/bash

echo "Module4: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make
# notice that the branchingimpact1 is built without passing -DARRAY_SIZE_X[=512]. This will build it with the maro ARRAY_SIZE_X defined as 32
# for the purpose of investigating the warp divergence due to branching

echo "----------------------------------------------------------------------------"
echo "Test 1: four previous math opertions using pageable host memory and device memory"
echo "----------------------------------------------------------------------------"


echo "recomplile with non-VERBOSE option to avoid fluiding the screen"

nvcc pageableHostmem_arithmetics_fake2D.cu -std=c++11 -DVERBOSE[=0] -L /usr/local/cuda/lib -lcudart -o pageableHostmem_arithmetics_fake2D

./pageableHostmem_arithmetics_fake2D 256 256

./pageableHostmem_arithmetics_fake2D 1024 256

./pageableHostmem_arithmetics_fake2D 32768 256

./pageableHostmem_arithmetics_fake2D 1048576 256


echo "-----------------------------------------------------------------------------------------------------------"
echo "Test 2: four previous math opertions using pinned host memory and device memory"
echo "-----------------------------------------------------------------------------------------------------------"



echo "recomplile with non-VERBOSE option to avoid fluiding the screen"

nvcc pinnedHostmem_arithmetics_fake2D.cu -std=c++11 -DVERBOSE[=0] -L /usr/local/cuda/lib -lcudart -o pinnedHostmem_arithmetics_fake2D
./pinnedHostmem_arithmetics_fake2D 256 256


./pinnedHostmem_arithmetics_fake2D 1024 256

./pinnedHostmem_arithmetics_fake2D 32768 256

./pinnedHostmem_arithmetics_fake2D 1048576 256

