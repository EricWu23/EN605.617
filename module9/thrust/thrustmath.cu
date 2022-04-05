
#include <stdio.h>
#include "utility.h"
#include "thrustmath.h"
#include "globalmacro.h"

/* Actual kernels that use the global device memory*/					
static void thrust_arrayAdd(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
static void thrust_arraySubtract(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
static void thrust_arrayMultiply(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);
static void thrust_arrayMod(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult);

void thrust_arrayAdd(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult) {
	for(int i=0;i<MAXOPERIONS;i++)
	{
    	thrust::transform(gpu_array0.begin(), gpu_array0.end(), gpu_arrayresult.begin(), gpu_arrayresult.begin(),thrust::plus<int>());
	}
}
void thrust_arraySubtract(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult) {
	for(int i=0;i<MAXOPERIONS;i++)
	{
    	thrust::transform(gpu_array0.begin(), gpu_array0.end(), gpu_arrayresult.begin(), gpu_arrayresult.begin(),thrust::minus<int>());
	}
}
void thrust_arrayMultiply(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult) {
	for(int i=0;i<MAXOPERIONS;i++)
	{
    	thrust::transform(gpu_array0.begin(), gpu_array0.end(), gpu_arrayresult.begin(), gpu_arrayresult.begin(),thrust::multiplies<int>());
	}
}
void thrust_arrayMod(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult) {
	for(int i=0;i<MAXOPERIONS;i++)
	{
    	thrust::transform(gpu_array0.begin(), gpu_array0.end(), gpu_arrayresult.begin(), gpu_arrayresult.begin(),thrust::modulus<int>());
	}
}

__host__ void execute_gpu_thrust_arrayAdd(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult){ 	

	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	thrust_arrayAdd(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 1-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	printf("GPU execution with thrust takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_thrust_arraySubtract(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult){ 	

	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	thrust_arraySubtract(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 1-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 1 (Subtract) is called! \n");printf("The Kernel 1 (Subtract) performs the math operation %d times! \n",MAXOPERIONS);
	printf("GPU execution with thrust takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}

__host__ void execute_gpu_thrust_arrayMultiply(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult){ 	

	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	thrust_arrayMultiply(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 1-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 2 (Multiply) is called! \n");printf("The Kernel 2 (Multiply) performs the math operation %d times! \n",MAXOPERIONS);
	printf("GPU execution with thrust takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}

__host__ void execute_gpu_thrust_arrayMod(thrust::device_vector<int> &gpu_array0,thrust::device_vector<int> &gpu_array1,thrust::device_vector<int> &gpu_arrayresult){ 	

	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	thrust_arrayMod(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 1-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 3 (Mod) is called! \n");printf("The Kernel 3 (Mod) performs the math operation %d times! \n",MAXOPERIONS);
	printf("GPU execution with thrust takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}



void execute_gpu_thrust_test(int numBlocks, int blockSize){
	printf("Unit Test 1: Simple Math Operations with thrust libary\n");
	printf("-------------------------------------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;

	thrust::host_vector<int> cpu_array0(totalThreads);
	thrust::host_vector<int> cpu_array1(totalThreads);
	thrust::host_vector<int> cpu_arrayresult(totalThreads);
    thrust::sequence(cpu_array0.begin(), cpu_array0.end(), 1);
	thrust::fill(cpu_array1.begin(), cpu_array1.end(), 75);
	thrust::fill(cpu_arrayresult.begin(), cpu_arrayresult.end(), 0);
    thrust::device_vector<int> gpu_array0=cpu_array0;
	thrust::device_vector<int> gpu_array1=cpu_array1;

	thrust::device_vector<int> gpu_arrayresult=cpu_arrayresult;
	for(int kernel=0; kernel<4; kernel++){//Execute 4 simple math operation
      switch(kernel){
            case 0:{ execute_gpu_thrust_arrayAdd(gpu_array0,gpu_array1, gpu_arrayresult);
                    } break;                                                                                     
            case 1:{ execute_gpu_thrust_arraySubtract(gpu_array0,gpu_array1, gpu_arrayresult);
                   }break;                                     
           case 2:{ execute_gpu_thrust_arrayMultiply(gpu_array0,gpu_array1, gpu_arrayresult);
                   }break;                                                                 
           case 3:{ execute_gpu_thrust_arrayMod(gpu_array0,gpu_array1, gpu_arrayresult);
                   }break;                                                                   
            default: exit(1); break;}	
	}	

	//cudaDeviceReset();//Destroy all allocations and reset all state on the current device in the current process
}