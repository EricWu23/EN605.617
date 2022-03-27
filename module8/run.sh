#!/bin/bash

echo "Module8: Tests started"

echo "Clean the folder by calling make clearn:"

make clean

echo "Call make to build necessary executable:"

make
# notice that the default operation for make will build a version with -DHA[=$(2)] -DWA[=$(9)] -DWB[=$(2)] -DHA2[=$(3)] -DWA2[=$(3)] -DNRHS[=$(1)]
# So matrix multiplication using 2x9 x 9x2 matrixes, and linear solver is solving a system equations of three equations.
# you can change the dimensions and/or system order though rebuild with -D options

echo "----------------------------------------------------------------------------------------------------------"
echo "Test 1: Matrix multiplication: 2x9 x 9x2 ---> 2x2;   Linear solve: 3x3 3x1"
echo "----------------------------------------------------------------------------------------------------------"

./assignment


echo "---------------------------------------------------------------------------------------------"
echo "Test 2: Matrix multiplication: 3x4 x 4x2 ---> 3x2;   Linear solve: 4x4 4x1 "
echo "---------------------------------------------------------------------------------------------"
 make HA=3 WA=4 WB=2 HA2=4 WA2=4 NRHS=1
./assignment

make clean