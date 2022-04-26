#!/bin/bash

echo "Module12: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make


echo "----------------------------------------------------------------------------------------------------------"
echo "Test 1: Testing moving average with window size 4 on 16 elements with zero padding "
echo "----------------------------------------------------------------------------------------------------------"

./assignment



