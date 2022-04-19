#!/bin/bash

echo "Module11: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make


echo "----------------------------------------------------------------------------------------------------------"
echo "Test 1: Testing OPENCL kernels Convolve (49x49 input, 7x7 mask)"
echo "----------------------------------------------------------------------------------------------------------"

./assignment



