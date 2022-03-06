//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include "assignment.h"

    
#define WARP 32
#define OFFSET 10
#define MAXARRAYSIZE 8192// this needs to be specify as multiple of blocksize
#ifndef MAXOPERIONS // this specifies how many operations to run inside a kernel
	#define MAXOPERIONS 100
#endif
#ifndef VERBOSE 
	#define VERBOSE 0
#endif
#define NUM_REGISTER_PER_THREAD 256
__constant__  static int constarray0[MAXARRAYSIZE];
__constant__  static int constarray1[MAXARRAYSIZE];

__device__ void copy_data_from_shared(int * const data,
									 int * const shared_tmp,
									const int globalid,
									const int tid)
{
	// deepcopy
	__syncthreads();// synchronize all the threads within a block
	data[globalid]=shared_tmp[tid];
}
__device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int globalid,
									const int tid)
{   // deepcopy
	shared_tmp[tid] = data[globalid];
	__syncthreads();// synchronize all the threads within a block
}

void cpu_array0_int(int* arr,int num_row,int num_column){			
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){		
				 arr[i*num_column+j]=i*num_column+j;
			}    
	 }				
}
void  cpu_array1_int(int* arr,int num_row,int num_column){		
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){
				 //arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
				 arr[i*num_column+j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
			}    	 
	 }				
}
void print_array(int* arr, int num_row, int num_col){
	printf("--------------------------------------------\n");
      for(int i=0; i<num_col; i++){
            for(int j=0; j<num_row; j++){
              if (i== num_col-1){
                  printf("%i\n", arr[j*num_col+i]);
              }
              else{
				  printf("%i ", arr[j*num_col+i]);
              }
            }
      }
      printf("--------------------------------------------\n");      
}


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


__global__ void gpu_arrayAdd_shared(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx; // collapse flat 2D down to 1D, whose index is global thread index
	 extern __shared__ int shared_tmp[];// total dynamically  allocated shared mem // 49KB limit
	 int *arry0shared=shared_tmp;
	 int *arry1shared=(int*)&shared_tmp[blockDim.x];
	 copy_data_to_shared(array0,arry0shared,global_idx,threadIdx.x);//array0-->arry0shared
	 copy_data_to_shared(array1,arry1shared,global_idx,threadIdx.x);//array1-->arry1shared
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++){
			arrayres[global_idx]=arry0shared[threadIdx.x]+arry1shared[threadIdx.x];
		}
    }
}
__global__ void gpu_arraySubtract_shared(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	

    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;    // collapse flat 2D down to 1D, whose index is global thread index
	 extern __shared__ int shared_tmp[];// total dynamically  allocated shared mem // 49KB limit
	 int *arry0shared=shared_tmp;
	 int *arry1shared=(int*)&shared_tmp[blockDim.x];
	 copy_data_to_shared(array0,arry0shared,global_idx,threadIdx.x);//array0-->arry0shared
	 copy_data_to_shared(array1,arry1shared,global_idx,threadIdx.x);//array1-->arry1shared
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=arry0shared[threadIdx.x]-arry1shared[threadIdx.x];
		}
    }

}    
__global__ void gpu_arrayMult_shared(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;    // collapse flat 2D down to 1D, whose index is global thread index
	 extern __shared__ int shared_tmp[];// total dynamically  allocated shared mem // 49KB limit
	 int *arry0shared=shared_tmp;
	 int *arry1shared=(int*)&shared_tmp[blockDim.x];
	 copy_data_to_shared(array0,arry0shared,global_idx,threadIdx.x);//array0-->arry0shared
	 copy_data_to_shared(array1,arry1shared,global_idx,threadIdx.x);//array1-->arry1shared
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=arry0shared[threadIdx.x]*arry1shared[threadIdx.x];
		}
    }
}
__global__ void gpu_arrayMod_shared(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;  // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;   // collapse flat 2D down to 1D, whose index is global thread index
	 extern __shared__ int shared_tmp[];// total dynamically  allocated shared mem // 49KB limit
	 int *arry0shared=shared_tmp;
	 int *arry1shared=(int*)&shared_tmp[blockDim.x];
	 copy_data_to_shared(array0,arry0shared,global_idx,threadIdx.x);//array0-->arry0shared
	 copy_data_to_shared(array1,arry1shared,global_idx,threadIdx.x);//array1-->arry1shared	 
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			arrayres[global_idx]=arry0shared[threadIdx.x]%arry1shared[threadIdx.x];
		}
    }
}

__global__ void gpu_arrayAdd_const(int* arrayres,int offset) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	   for(int i=0;i<MAXOPERIONS;i++){
		 arrayres[offset+global_idx]=constarray0[global_idx]+constarray1[global_idx];	
	   }
	}	
}
__global__ void gpu_arraySubtruct_const(int* arrayres,int offset) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;	// collapse flat 2D down to 1D, whose index is global thread index
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	   for(int i=0;i<MAXOPERIONS;i++){
		 arrayres[offset+global_idx]=constarray0[global_idx]-constarray1[global_idx];	
	   }
	}	
}
__global__ void gpu_arrayMult_const(int* arrayres,int offset) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;    // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;    // collapse flat 2D down to 1D, whose index is global thread index
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	   for(int i=0;i<MAXOPERIONS;i++){
		 arrayres[offset+global_idx]=constarray0[global_idx]*constarray1[global_idx];	
	   }
	}	
}
__global__ void gpu_arrayMod_const(int* arrayres,int offset) {   
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x; // collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;    // collapse flat 2D down to 1D, whose index is global thread index
	if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	   for(int i=0;i<MAXOPERIONS;i++){
		 arrayres[offset+global_idx]=constarray0[global_idx]%constarray1[global_idx];	
	   }
	}	
}

__global__ void gpu_arrayAdd_register(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index
	//int registermem0[NUM_REGISTER_PER_THREAD/2];
	//int registermem1[NUM_REGISTER_PER_THREAD/2];
	//registermem0[global_idx%(NUM_REGISTER_PER_THREAD/2)]=array0[global_idx];
	//registermem1[global_idx%(NUM_REGISTER_PER_THREAD/2)]=array0[global_idx];
	int registermem0=array0[global_idx];
	int registermem1=array1[global_idx];
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++)
		{
			//arrayres[global_idx]=registermem0[global_idx%(NUM_REGISTER_PER_THREAD/2)]+registermem1[global_idx%(NUM_REGISTER_PER_THREAD/2)];
			arrayres[global_idx]=registermem0+registermem1;
		}
    }
}
__global__ void gpu_arraySubtract_register(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index
	int registermem0=array0[global_idx];
	int registermem1=array1[global_idx];
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++){
			arrayres[global_idx]=registermem0-registermem1;
		}
    }
}
__global__ void gpu_arrayMult_register(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index
	int registermem0=array0[global_idx];
	int registermem1=array1[global_idx];
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++){
			arrayres[global_idx]=registermem0*registermem1;
		}
    }
}
__global__ void gpu_arrayMod_register(int *array0,int *array1,int* arrayres) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;// collapse the higher dimension layout or nested layout down to flat 2D
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;// collapse flat 2D down to 1D, whose index is global thread index
	int registermem0=array0[global_idx];
	int registermem1=array1[global_idx];
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
	    for(int i=0;i<MAXOPERIONS;i++){
			arrayres[global_idx]=registermem0%registermem1;
		}
    }
}

__host__ void execute_gpu_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){ 	
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
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	arrayAdd<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
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
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	arraySubtract<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 1 (subtract) is called! \n");printf("The Kernel 1 (subtract) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
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
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	arrayMult<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 2 (multiplication) is called! \n");printf("Kernel 2 (multiplication) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); }
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}
__host__ void execute_gpu_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
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
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	arrayMod<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);						
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu 
	printf("Kernel 3 (mod) is called! \n");printf("Kernel 3 (mod) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}// debug only
	printf("GPU execution with global mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}

__host__ void execute_gpu_sharedmem_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // using 1 row, blockSize colmns layout 
    const dim3 blocks_layout(numBlocks,1);// there are multiple ways of layout to achieve numBlocks, I choose to fix the  1 row, numBlocks colmn layout
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayAdd_shared<<<blocks_layout,threads_layout,blockSize*3*sizeof(int)>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with shared mem takes: %.3fms\n",delta_time1);printf("*******\n");;
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_sharedmem_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // 
    const dim3 blocks_layout(numBlocks,1);// 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arraySubtract_shared<<<blocks_layout,threads_layout,blockSize*2*sizeof(int)>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 1 (subtract) is called! \n");printf("The Kernel 1 (subtract) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 }
	printf("GPU execution with shared mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_sharedmem_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // 
    const dim3 blocks_layout(numBlocks,1);// 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayMult_shared<<<blocks_layout,threads_layout,blockSize*2*sizeof(int)>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 2 (multiplication) is called! \n");printf("Kernel 2 (multiplication) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); }//debug only
	printf("GPU execution with shared mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}
__host__ void execute_gpu_sharedmem_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); 
    const dim3 blocks_layout(numBlocks,1);
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayMod_shared<<<blocks_layout,threads_layout,blockSize*2*sizeof(int)>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);						
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); 
	printf("Kernel 3 (mod) is called! \n");printf("Kernel 3 (mod) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}
	printf("GPU execution with shared mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}

__host__ void execute_gpu_constmem_arrayAdd(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); 
    const dim3 blocks_layout(numBlocks,1); 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1,delta_time1_sum = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	int offset=0;
	/* copy from host to const memory on Device and execute kernel*/
	if(totalThreads<=(int)MAXARRAYSIZE){
		cudaMemcpyToSymbol(constarray0, cpu_array0,(size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(constarray1, cpu_array1, (size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		offset=0;
	    cudaEventRecord(kernel_start1, 0);//0 is the default stream
		gpu_arrayAdd_const<<<blocks_layout,threads_layout>>>(gpu_arrayresult,offset);
		cudaEventRecord(kernel_stop1, 0);
		cudaEventSynchronize(kernel_stop1);
		cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
		delta_time1_sum=delta_time1;
	}
	else{// totalThreads>MAXARRAYSIZE		
		for(offset=0;offset<(int)totalThreads;offset+=(int)MAXARRAYSIZE){
			cudaMemcpyToSymbol(constarray0,&cpu_array0[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol(constarray1,&cpu_array1[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaEventRecord(kernel_start1, 0);//0 is the default stream
			gpu_arrayAdd_const<<<MAXARRAYSIZE/blockSize,blockSize>>>(gpu_arrayresult,offset);
			cudaEventRecord(kernel_stop1, 0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
			delta_time1_sum+=delta_time1;			
		}
		printf("delta_time1_sum: %.3f\n",delta_time1_sum);
	}
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost);
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}
	printf("GPU execution with const mem takes: %.3fms\n",delta_time1_sum);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_constmem_arraySubtract(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1,delta_time1_sum = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	int offset=0;
	/* copy from host to const memory on Device and execute kernel*/
	if(totalThreads<=(int)MAXARRAYSIZE){
		printf("totalThreads: %d\n",totalThreads);
		cudaMemcpyToSymbol(constarray0, cpu_array0,(size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(constarray1, cpu_array1, (size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		offset=0;
	    cudaEventRecord(kernel_start1, 0);//0 is the default stream
		gpu_arraySubtruct_const<<<blocks_layout,threads_layout>>>(gpu_arrayresult,offset);
		cudaEventRecord(kernel_stop1, 0);
		cudaEventSynchronize(kernel_stop1);
		cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
		delta_time1_sum=delta_time1;
	}
	else{// totalThreads>MAXARRAYSIZE
		for(offset=0;offset<(int)totalThreads;offset+=(int)MAXARRAYSIZE){
			cudaMemcpyToSymbol(constarray0,&cpu_array0[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol(constarray1,&cpu_array1[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaEventRecord(kernel_start1, 0);//0 is the default stream
			gpu_arraySubtruct_const<<<MAXARRAYSIZE/blockSize,blockSize>>>(gpu_arrayresult,offset);
			cudaEventRecord(kernel_stop1, 0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
			delta_time1_sum+=delta_time1;			
		}
		printf("delta_time1_sum: %.3f\n",delta_time1_sum);
	}
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 1 (subtract) is called! \n");printf("The Kernel 1 (subtract) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with const mem takes: %.3fms\n",delta_time1_sum);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_constmem_arrayMult(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1,delta_time1_sum = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	int offset=0;
	/* copy from host to const memory on Device and execute kernel*/
	if(totalThreads<=(int)MAXARRAYSIZE){
		cudaMemcpyToSymbol(constarray0, cpu_array0,(size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(constarray1, cpu_array1, (size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		offset=0;
	    cudaEventRecord(kernel_start1, 0);//0 is the default stream
		gpu_arrayMult_const<<<blocks_layout,threads_layout>>>(gpu_arrayresult,offset);
		cudaEventRecord(kernel_stop1, 0);
		cudaEventSynchronize(kernel_stop1);
		cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
		delta_time1_sum=delta_time1;
	}
	else{// totalThreads>MAXARRAYSIZE
		for(offset=0;offset<(int)totalThreads;offset+=(int)MAXARRAYSIZE){
			cudaMemcpyToSymbol(constarray0,&cpu_array0[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol(constarray1,&cpu_array1[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaEventRecord(kernel_start1, 0);//0 is the default stream
			gpu_arrayMult_const<<<MAXARRAYSIZE/blockSize,blockSize>>>(gpu_arrayresult,offset);
			cudaEventRecord(kernel_stop1, 0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
			delta_time1_sum+=delta_time1;			
		}
	}
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 2 (multiplication) is called! \n");printf("Kernel 2 (multiplication) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}// debug only
	printf("GPU execution with const mem takes: %.3fms\n",delta_time1_sum);printf("*******\n");
	cudaEventDestroy(kernel_start1);
	cudaEventDestroy(kernel_stop1);	
}
__host__ void execute_gpu_constmem_arrayMod(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1,delta_time1_sum = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	int offset=0;
	/* copy from host to const memory on Device and execute kernel*/
	if(totalThreads<=(int)MAXARRAYSIZE){
		cudaMemcpyToSymbol(constarray0, cpu_array0,(size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(constarray1, cpu_array1, (size_t)sizeof(int)*totalThreads,0,cudaMemcpyHostToDevice);
		offset=0;
	    cudaEventRecord(kernel_start1, 0);//0 is the default stream
		gpu_arrayMod_const<<<blocks_layout,threads_layout>>>(gpu_arrayresult,offset);
		cudaEventRecord(kernel_stop1, 0);
		cudaEventSynchronize(kernel_stop1);
		cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
		delta_time1_sum=delta_time1;
	}
	else{// totalThreads>MAXARRAYSIZE
		for(offset=0;offset<(int)totalThreads;offset+=(int)MAXARRAYSIZE){
			cudaMemcpyToSymbol(constarray0,&cpu_array0[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaMemcpyToSymbol(constarray1,&cpu_array1[offset], sizeof(int)*MAXARRAYSIZE,0,cudaMemcpyHostToDevice );
			cudaEventRecord(kernel_start1, 0);//0 is the default stream
			gpu_arrayMod_const<<<MAXARRAYSIZE/blockSize,blockSize>>>(gpu_arrayresult,offset);
			cudaEventRecord(kernel_stop1, 0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
			delta_time1_sum+=delta_time1;			
		}
	}
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 3 (mod) is called! \n");printf("Kernel 3 (mod) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with const mem takes: %.3fms\n",delta_time1_sum);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}

__host__ void execute_gpu_register_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // using 1 row, blockSize colmns layout 
    const dim3 blocks_layout(numBlocks,1);// there are multiple ways of layout to achieve numBlocks, I choose to fix the  1 row, numBlocks colmn layout
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayAdd_register<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 0 (Add) is called! \n");printf("The Kernel 0 (Add) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	printf("GPU execution with register mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_register_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // 
    const dim3 blocks_layout(numBlocks,1);// 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arraySubtract_register<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 1 (subtract) is called! \n");printf("The Kernel 1 (subtract) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 }
	printf("GPU execution with register mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_register_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // 
    const dim3 blocks_layout(numBlocks,1);// 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayMult_register<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 2 (multiplication) is called! \n");printf("Kernel 2 (multiplication) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); }//debug only
	printf("GPU execution with register mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);
}
__host__ void execute_gpu_register_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	const dim3 threads_layout(blockSize,1); // 
    const dim3 blocks_layout(numBlocks,1);// 
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	gpu_arrayMod_register<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	cudaEventRecord(kernel_stop1, 0);//0 is the default stream
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);						
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); 
	printf("Kernel 3 (mod) is called! \n");printf("Kernel 3 (mod) performs the math operation %d times! \n",MAXOPERIONS);
	if(VERBOSE){printf("Array Result:\n");print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);}
	printf("GPU execution with register mem takes: %.3fms\n",delta_time1);printf("*******\n");
	cudaEventDestroy(kernel_start1);cudaEventDestroy(kernel_stop1);	
}


void execute_gpu_global_test(int numBlocks, int blockSize){
	printf("Unit Test 1: Simple Math Operations with global memory\n");
	printf("-------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){//print out the arrays for debuging
		printf("The following two arrays are initialized on cpu! \n");
		printf("Array0:\n");print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice ); 
	for(int kernel=0; kernel<4; kernel++){//Execute 4 simple math operation
      switch(kernel){
            case 0:{ execute_gpu_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                    } break;                                                                                     
            case 1:{execute_gpu_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                     
           case 2:{execute_gpu_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                 
           case 3:{ execute_gpu_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                   
            default: exit(1); break;}	
	}	
	free(cpu_array0);free(cpu_array1);free(cpu_array_res);
    cudaFree(gpu_array0);cudaFree(gpu_array1);cudaFree(gpu_arrayresult);	
	cudaDeviceReset();//Destroy all allocations and reset all state on the current device in the current process
}
void execute_gpu_shared_test(int numBlocks, int blockSize){
    printf("Unit Test2: Simple Math Operations with shared memory\n");
    printf("-------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	/* dynamically allocate the memory on the host*/
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){//print out the arrays for debuging
		printf("The following two arrays are initialized on cpu! \n");
		printf("Array0:\n");print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
	 /* Device memory allocation */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
	for(int kernel=0; kernel<4; kernel++){//Execute 4 simple math operation 
      switch(kernel){
            case 0:{ execute_gpu_sharedmem_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                    } break;                                                                                     
            case 1:{execute_gpu_sharedmem_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                     
           case 2:{execute_gpu_sharedmem_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                 
           case 3:{execute_gpu_sharedmem_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                   
            default: exit(1); break;                                                                                                         
      }	
	}	
	free(cpu_array0);
	free(cpu_array1);
	free(cpu_array_res);
    cudaFree(gpu_array0);
	cudaFree(gpu_array1);
    cudaFree(gpu_arrayresult);	
	cudaDeviceReset();	//Destroy all allocations and reset all state on the current device in the current process
}
void execute_gpu_const_test(int numBlocks, int blockSize){
	printf("Unit Test3: Simple Math Operations with constant memory\n");
    printf("-------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	/* dynamically allocate the memory on the host*/
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	/* print out the arrays for debuging */
	if(VERBOSE){
		printf("The following two arrays are initialized on cpu! \n");printf("Array0:\n");print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
	 /* Device memory allocation */
	int *gpu_arrayresult;
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	/* Execute 4 simple math operation*/ 
	for(int kernel=0; kernel<4; kernel++){
      switch(kernel){
            case 0:{ execute_gpu_constmem_arrayAdd(numBlocks,blockSize,cpu_array0,cpu_array1,gpu_arrayresult,cpu_array_res);  
                    } break;                                                                                     
            case 1:{ execute_gpu_constmem_arraySubtract(numBlocks,blockSize,cpu_array0,cpu_array1,gpu_arrayresult,cpu_array_res); 
                   }break;                                     
           case 2:{execute_gpu_constmem_arrayMult(numBlocks,blockSize,cpu_array0,cpu_array1,gpu_arrayresult,cpu_array_res);
				   }break;                                                                 
           case 3:{ execute_gpu_constmem_arrayMod(numBlocks,blockSize,cpu_array0,cpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                   
            default: exit(1); break;                                                                                                         
      }	
	}
	/*Free Allocated memories*/
	free(cpu_array0);
	free(cpu_array1);
	free(cpu_array_res);
    cudaFree(gpu_arrayresult);	
	cudaDeviceReset();	//Destroy all allocations and reset all state on the current device in the current process
}
void execute_gpu_register_test(int numBlocks, int blockSize){
	printf("Unit Test4: Simple Math Operations with register memory\n");
    printf("-------------------------------------------------------\n");
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	/* dynamically allocate the memory on the host*/
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	if(VERBOSE){printf("The following two arrays are initialized on cpu! \n");printf("Array0:\n");
	print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);printf("Array1:\n");print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);}//debug only
	 /* Device memory allocation */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
	for(int kernel=0; kernel<4; kernel++){//Execute 4 simple math operation 
      switch(kernel){
            case 0:{ execute_gpu_register_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                    } break;                                                                                     
            case 1:{ execute_gpu_register_arraySubtract(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                     
           case 2:{ execute_gpu_register_arrayMult(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                 
           case 3:{execute_gpu_register_arrayMod(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                   }break;                                                                   
            default: exit(1); break;                                                                                                         
      }	
	}	
	free(cpu_array0);free(cpu_array1);free(cpu_array_res);
    cudaFree(gpu_array0);cudaFree(gpu_array1);cudaFree(gpu_arrayresult);	
	cudaDeviceReset();	//Destroy all allocations and reset all state on the current device in the current process
}

int main(int argc, char** argv)
{
	int totalThreads = (1 << 20);
	int blockSize = 256;
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	int numBlocks = totalThreads/blockSize;
    if(blockSize<WARP){ 	// code check to make sure blockSize is multiple of WARP                      
           blockSize=WARP;
           printf("Warning: Block size specified is less than size of WARP.It got modified to be: %i\n",WARP);     
    }
    else{
            if(blockSize % WARP!=0){
                    blockSize=(blockSize+0.5*WARP)/WARP*WARP;
                    printf("Warning: Block size specified is not evenly divisible by the size of WARP.\n");
                    printf("It got modified to be the nearst number that can be evenly divisible by the size of WARP.\n");
					printf("Now, the blocksize is:%i\n",blockSize);     
            }
    }
	if (totalThreads % blockSize != 0) {	// code check to make sure Total number of threads is multiple of blockSize
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	execute_gpu_global_test(numBlocks,blockSize); //  test harness for executing kernel using global memory
	execute_gpu_shared_test(numBlocks,blockSize); //  test harness for executing kernel using shared memory
	execute_gpu_const_test(numBlocks,blockSize);  //  test harness for executing kernel using constant memory
	execute_gpu_register_test(numBlocks,blockSize);  //  test harness for executing kernel using register memory
}
