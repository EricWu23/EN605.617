#include <stdio.h>
#include "stream.h"
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

__host__ void execute_gpu_stream_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1){ 	
    int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1; 
    cudaStreamCreate(&stream1); 
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream1); 	
	arrayAdd<<<blocks_layout,threads_layout,0,stream1>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays 
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream1); // memcopy from gpu to cpu
	cudaStreamSynchronize(stream1);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(h_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)
}
__host__ void execute_gpu_stream_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1; 
    cudaStreamCreate(&stream1);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	arraySubtract<<<blocks_layout,threads_layout,0,stream1>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream1);
	cudaStreamSynchronize(stream1);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 1 (subtract) is called! \n");printf("The Kernel 1 (subtract) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(h_array_res,cpu_arr_size_y,cpu_arr_size_x);}
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)
}
__host__ void execute_gpu_stream_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1){
    int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1; 
    cudaStreamCreate(&stream1);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream1); 
	arrayMult<<<blocks_layout,threads_layout,0,stream1>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream1); // memcopy from gpu to cpu
	cudaStreamSynchronize(stream1);// host block until all tasks on stream1 finish
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	printf("Kernel 2 (multiplication) is called! \n");printf("Kernel 2 (multiplication) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(h_array_res,cpu_arr_size_y,cpu_arr_size_x); }
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)	
}
__host__ void execute_gpu_stream_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1){
    int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaStream_t stream1; 
    cudaStreamCreate(&stream1);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	cudaMemcpyAsync(gpu_array0,h_array0,size_in_bytes,cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(gpu_array1,h_array1,size_in_bytes,cudaMemcpyHostToDevice, stream1); 
	arrayMod<<<blocks_layout,threads_layout,0,stream1>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays
	cudaMemcpyAsync(h_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost,stream1); // memcopy from gpu to cpu
	cudaStreamSynchronize(stream1);// host block until all tasks on stream1 finish
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);						
	printf("Kernel 3 (mod) is called! \n");printf("Kernel 3 (mod) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(h_array_res,cpu_arr_size_y,cpu_arr_size_x);}// debug only
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)	
}

void execute_gpu_stream_test(int numBlocks, int blockSize){
	printf("Unit Test 1: Simple Math Operations with global device memory using Streams\n");
	printf("---------------------------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	int *h_array0,*h_array1,*h_array_res;
	cudaMallocHost(&h_array0,size_in_bytes);//pinned 
	cudaMallocHost(&h_array1,size_in_bytes);
	cudaMallocHost(&h_array_res,size_in_bytes);	
    cpu_array0_int(h_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(h_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){//print out the arrays for debuging
		printf("The following two arrays are initialized on cpu! \n");
		printf("Array0:\n");print_array(h_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");print_array(h_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	for(int kernel=0; kernel<4; kernel++){//Execute 4 simple math operation
      switch(kernel){
            case 0:{ execute_gpu_stream_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,h_array_res,h_array0,h_array1);
                    } break;                                                                                     
            case 1:{execute_gpu_stream_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,h_array_res,h_array0,h_array1);
                   }break;                                     
           case 2:{execute_gpu_stream_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,h_array_res,h_array0,h_array1);
                   }break;                                                                 
           case 3:{ execute_gpu_stream_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,h_array_res,h_array0,h_array1);
                   }break;                                                                   
            default: exit(1); break;}	
	}	
	cudaFreeHost(h_array0);cudaFreeHost(h_array1);cudaFreeHost(h_array_res);
    cudaFree(gpu_array0);cudaFree(gpu_array1);cudaFree(gpu_arrayresult);	
	cudaDeviceReset();//Destroy all allocations and reset all state on the current device in the current process
}