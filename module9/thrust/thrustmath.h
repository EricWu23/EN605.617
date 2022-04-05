#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
/*test harness to test kernels that uses thrust library*/ 
extern void execute_gpu_thrust_arrayAdd(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
extern void execute_gpu_thrust_arraySubtract(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
extern void execute_gpu_thrust_arrayMultiply(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
extern void execute_gpu_thrust_arrayMod(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
/* highest level test harness that run all math operations*/
extern void execute_gpu_thrust_test(int numBlocks, int blockSize);