#!/bin/bash

echo "Module10: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make


echo "----------------------------------------------------------------------------------------------------------"
echo "Test 1: Testing Five mathmetical OPENCL kernels "
echo "----------------------------------------------------------------------------------------------------------"

./assignment



