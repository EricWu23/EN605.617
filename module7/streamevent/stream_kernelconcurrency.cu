#include <stdio.h>
#include "stream_kernelconcurrency.h"
#include "utility.h"
#include "globalmacro.h"

/* Actual kernels that use the global device memory*/					
static __global__ void arrayAdd(int *array0,int *array1,int* arrayres);
static __global__ void arraySubtract(int *array0,int *array1,int* arrayres);
static __global__ void arrayMult(int *array0,int *array1,int* arrayres);
static __global__ void arrayMod(int *array0,int *array1,int* arrayres);

__global__ void arrayAdd(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index   
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=array0[global_idx]+array1[global_idx];
		}
    }
}
__global__ void arraySubtract(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x; // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx; // collapse flat 2D down to 1D, whose index is global thread index

	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
		for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=array0[global_idx]-array1[global_idx];
		}
	}
}
__global__ void arrayMult(int *array0,int *array1,int* arrayres) {

    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;


    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;    // collapse flat 2D down to 1D, whose index is global thread index
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=array0[global_idx]*array1[global_idx];
		}
    }
}
__global__ void arrayMod(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;    // collapse flat 2D down to 1D, whose index is global thread index
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=array0[global_idx]%array1[global_idx];
		}
    }
}


__host__ void execute_gpu_streamkernelconcurrency_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	arrayAdd<<<blocks_layout,threads_layout,0,stream>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream);
}

__host__ void execute_gpu_streamkernelconcurrency_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	arraySubtract<<<blocks_layout,threads_layout,0,stream>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream);
}
__host__ void execute_gpu_streamkernelconcurrency_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	arrayMult<<<blocks_layout,threads_layout,0,stream>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream);
}

__host__ void execute_gpu_streamkernelconcurrency_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	arrayMod<<<blocks_layout,threads_layout,0,stream>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream);
}


__host__ void execute_gpu_streamkernelconcurrency_test(int numBlocks, int blockSize){
	printf("Unit Test 1: Simple Math Operations with global device memory using Streams with kernel concurrencies\n");
	printf("---------------------------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	int *h_array0,*h_array1,*h_array_res0,*h_array_res1,*h_array_res2,*h_array_res3;
	cudaMallocHost(&h_array0,size_in_bytes); cudaMallocHost(&h_array1,size_in_bytes);
	cudaMallocHost(&h_array_res0,size_in_bytes);cudaMallocHost(&h_array_res1,size_in_bytes);
    cudaMallocHost(&h_array_res2,size_in_bytes);cudaMallocHost(&h_array_res3,size_in_bytes);	
    cpu_array0_int(h_array0,cpu_arr_size_y,cpu_arr_size_x);cpu_array1_int(h_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){//print out the arrays just for debuging
		printf("The following two arrays are initialized on cpu! \n");printf("Array0:\n");print_array(h_array0,cpu_arr_size_y,cpu_arr_size_x);printf("Array1:\n");print_array(h_array1,cpu_arr_size_y,cpu_arr_size_x);}//just for debugging
    int * gpu_array0, * gpu_array1,*gpu_arrayresult0,*gpu_arrayresult1,*gpu_arrayresult2,*gpu_arrayresult3;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult0, size_in_bytes);cudaMalloc((void **)&gpu_arrayresult1, size_in_bytes);
	cudaMalloc((void **)&gpu_arrayresult2, size_in_bytes);cudaMalloc((void **)&gpu_arrayresult3, size_in_bytes);
	cudaEvent_t kernel_start1, kernel_stop1;float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1,stream2,stream3,stream4; 
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);cudaStreamCreate(&stream3); cudaStreamCreate(&stream4);
	cudaEventRecord(kernel_start1, stream1);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream2); 
	cudaStreamSynchronize(stream1);cudaStreamSynchronize(stream2);//block the host until data copy is complete
	execute_gpu_streamkernelconcurrency_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult0,h_array_res0,stream1);
	execute_gpu_streamkernelconcurrency_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult1,h_array_res1,stream2);
	execute_gpu_streamkernelconcurrency_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult2,h_array_res2,stream3);
	execute_gpu_streamkernelconcurrency_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult3,h_array_res3,stream4);
	cudaStreamSynchronize(stream1);cudaStreamSynchronize(stream2);cudaStreamSynchronize(stream3);cudaStreamSynchronize(stream4);// host block until all tasks on stream1,2,3,4 finish
	cudaEventRecord(kernel_stop1, stream1);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Total GPU execution and memory copy took: %.3fms\n",delta_time1);
	cudaFreeHost(h_array0);cudaFreeHost(h_array1);cudaFreeHost(h_array_res0);cudaFreeHost(h_array_res1);cudaFreeHost(h_array_res2);cudaFreeHost(h_array_res3);
    cudaFree(gpu_array0);cudaFree(gpu_array1);
	cudaFree(gpu_arrayresult0);cudaFree(gpu_arrayresult1);cudaFree(gpu_arrayresult2);cudaFree(gpu_arrayresult3);
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
	cudaStreamDestroy(stream1);cudaStreamDestroy(stream2);cudaStreamDestroy(stream3);cudaStreamDestroy(stream4);				
	cudaDeviceReset();//Destroy all allocations and reset all state on the current device in the current process
}

__host__ void execute_gpu_streamNokernelconcurrency_test(int numBlocks, int blockSize){
	printf("Unit Test 1: Simple Math Operations with global device memory using Streams without kernel concurrencies\n");
	printf("---------------------------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;int cpu_arr_size_x=totalThreads;
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	int *h_array0,*h_array1,*h_array_res0,*h_array_res1,*h_array_res2,*h_array_res3;
	cudaMallocHost(&h_array0,size_in_bytes); cudaMallocHost(&h_array1,size_in_bytes);
	cudaMallocHost(&h_array_res0,size_in_bytes);cudaMallocHost(&h_array_res1,size_in_bytes);
    cudaMallocHost(&h_array_res2,size_in_bytes);cudaMallocHost(&h_array_res3,size_in_bytes);	
    cpu_array0_int(h_array0,cpu_arr_size_y,cpu_arr_size_x);cpu_array1_int(h_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){//print out the arrays just for debuging
		printf("The following two arrays are initialized on cpu! \n");printf("Array0:\n");print_array(h_array0,cpu_arr_size_y,cpu_arr_size_x);printf("Array1:\n");print_array(h_array1,cpu_arr_size_y,cpu_arr_size_x);}//just for debugging
    int * gpu_array0, * gpu_array1,*gpu_arrayresult0,*gpu_arrayresult1,*gpu_arrayresult2,*gpu_arrayresult3;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult0, size_in_bytes);cudaMalloc((void **)&gpu_arrayresult1, size_in_bytes);
	cudaMalloc((void **)&gpu_arrayresult2, size_in_bytes);cudaMalloc((void **)&gpu_arrayresult3, size_in_bytes);
	cudaEvent_t kernel_start1, kernel_stop1;float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1,stream2; 
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
	cudaEventRecord(kernel_start1, stream1);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream2); 
	cudaStreamSynchronize(stream1);cudaStreamSynchronize(stream2);//block the host until data copy is complete
	execute_gpu_streamkernelconcurrency_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult0,h_array_res0,stream1);
	execute_gpu_streamkernelconcurrency_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult1,h_array_res1,stream1);
	execute_gpu_streamkernelconcurrency_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult2,h_array_res2,stream1);
	execute_gpu_streamkernelconcurrency_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult3,h_array_res3,stream1);
	cudaStreamSynchronize(stream1);cudaStreamSynchronize(stream2);
	cudaEventRecord(kernel_stop1, stream1);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Total GPU execution and memory copy took: %.3fms\n",delta_time1);
	cudaFreeHost(h_array0);cudaFreeHost(h_array1);cudaFreeHost(h_array_res0);cudaFreeHost(h_array_res1);cudaFreeHost(h_array_res2);cudaFreeHost(h_array_res3);
    cudaFree(gpu_array0);cudaFree(gpu_array1);
	cudaFree(gpu_arrayresult0);cudaFree(gpu_arrayresult1);cudaFree(gpu_arrayresult2);cudaFree(gpu_arrayresult3);
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
	cudaStreamDestroy(stream1);cudaStreamDestroy(stream2);				
	cudaDeviceReset();//Destroy all allocations and reset all state on the current device in the current process
}