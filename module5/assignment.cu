//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "assignment.h"
    
#define WARP 32
#define OFFSET 10
#ifndef VERBOSE 
	#define VERBOSE 1
#endif

__global__ void arrayAdd(int *array0,int *array1,int* arraysum) {

    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    // collapse flat 2D down to 1D, whose index is global thread index
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
    
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
        arraysum[global_idx]=array0[global_idx]+array1[global_idx];
    }
}

__global__ void arraySubtract(int *array0,int *array1,int* arraysub) {

    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    // collapse flat 2D down to 1D, whose index is global thread index
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
    
    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
        arraysub[global_idx]=array0[global_idx]-array1[global_idx];
    }
}

__global__ void arrayMult(int *array0,int *array1,int* arraymult) {

    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // collapse flat 2D down to 1D, whose index is global thread index
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
    arraymult[global_idx]=array0[global_idx]*array1[global_idx];
    }
}

__global__ void arrayMod(int *array0,int *array1,int* arraymod) {

    // collapse the higher dimension layout or nested layout down to flat 2D
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // collapse flat 2D down to 1D, whose index is global thread index
    const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    if(idx<(gridDim.x*blockDim.x) && idy<(gridDim.y*blockDim.y)){
    arraymod[global_idx]=array0[global_idx]%array1[global_idx];
    }
}    

__host__ void execute_gpu_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	
	/* layout specification*/
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	
	cudaEvent_t kernel_start1, kernel_stop1;
	float delta_time1 = 0.0f;
	cudaEventCreate(&kernel_start1);
	cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	
	
	//record events around kernel launch
	cudaEventRecord(kernel_start1, 0);//0 is the default stream
	arrayAdd<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays 
	cudaEventRecord(kernel_stop1, 0);
	cudaEventSynchronize(kernel_stop1);
	cudaEventElapsedTime(&delta_time1, kernel_start1,kernel_stop1);	
	
	cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel 0 (Add) is called! \n");
	if(VERBOSE)
	{
		printf("Array Result:\n");
		print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 
	}
	printf("\n GPU execution with global mem takes: %.3fms",delta_time1);
	printf("--------------------------------------------\n");
	
	cudaEventDestroy(kernel_start1);
	cudaEventDestroy(kernel_stop1);
}
// function to print out a 2D array for debugging
void print_array(int* arr, int num_row, int num_col)
{
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


/* initialize the data in the array according to assignment requirement*/
void cpu_array0_int(int* arr,int num_row,int num_column){		
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){		
				 arr[i*num_column+j]=i*num_column+j;
			}    
	 }				
}


/* initialize the data in the array according to assignment requirement*/
void  cpu_array1_int(int* arr,int num_row,int num_column){	
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){
				 //arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
				 arr[i*num_column+j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
			}    	 
	 }				
}


void execute_gpu_global_test(int numBlocks, int blockSize){
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
	int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);

	/* dynamically allocate the memory on the host*/
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
	/* data init*/
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	/* print out the arrays for debuging */
	if(VERBOSE){
		printf("The following two arrays are initialized on cpu! \n");
		printf("Array0:\n");
		print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");
		print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
	/* layout specification*/
	const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
	
	 /* Device memory allocation */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
	cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
	/* explicit memory copy from cpu to device*/
	cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
	//create events for timing the kernel executions.
	//cudaEvent_t kernel_start1, kernel_stop1,kernel_start2, kernel_stop2,kernel_start3, kernel_stop3,kernel_start4, kernel_stop4;
	cudaEvent_t kernel_start2, kernel_stop2,kernel_start3, kernel_stop3,kernel_start4, kernel_stop4;
	//float delta_time1 = 0.0f, delta_time2 = 0.0F,delta_time3 = 0.0F,delta_time4 = 0.0F;
	float delta_time2 = 0.0F,delta_time3 = 0.0F,delta_time4 = 0.0F;
	//cudaEventCreate(&kernel_start1);
	cudaEventCreate(&kernel_start2);
	cudaEventCreate(&kernel_start3);
	cudaEventCreate(&kernel_start4);
	//cudaEventCreateWithFlags(&kernel_stop1,cudaEventBlockingSync);
	cudaEventCreateWithFlags(&kernel_stop2,cudaEventBlockingSync);
	cudaEventCreateWithFlags(&kernel_stop3,cudaEventBlockingSync);
	cudaEventCreateWithFlags(&kernel_stop4,cudaEventBlockingSync);
	
	
	/* Execute 4 simple math operation*/ 
	for(int kernel=0; kernel<4; kernel++)
    {
      switch(kernel)
      {
            case 0:{ 
					  execute_gpu_arrayAdd(numBlocks,blockSize,gpu_array0,gpu_array1,gpu_arrayresult,cpu_array_res);
                    } break;                                                                                     
            case 1:{
						cudaEventRecord(kernel_start2, 0);//0 is the default stream
						arraySubtract<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
						cudaEventRecord(kernel_stop2, 0);
						cudaEventSynchronize(kernel_stop2);
						cudaEventElapsedTime(&delta_time2, kernel_start2,kernel_stop2);
						cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
						printf("Kernel 1 (subtract) is called! \n");
						if(VERBOSE){
							printf("Array Result:\n");
							print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 
						 }
						printf("\n GPU execution with global mem takes: %.3fms",delta_time2);
                        printf("--------------------------------------------\n");
                   }break;                                     
           case 2:{
						cudaEventRecord(kernel_start3, 0);//0 is the default stream
						arrayMult<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
						cudaEventRecord(kernel_stop3, 0);//0 is the default stream
						cudaEventSynchronize(kernel_stop3);
						cudaEventElapsedTime(&delta_time3, kernel_start3,kernel_stop3);
						cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
						printf("Kernel 2 (multiplication) is called! \n");
						if(VERBOSE){
							 printf("Array Result:\n");
							 print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); 
						}
						printf("\n GPU execution with global mem takes: %.3fms",delta_time3);
						printf("--------------------------------------------\n");
                   }break;                                                                 
           case 3:{      
						cudaEventRecord(kernel_start4, 0);//0 is the default stream
						arrayMod<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays
						cudaEventRecord(kernel_stop4, 0);//0 is the default stream
						cudaEventSynchronize(kernel_stop4);
						cudaEventElapsedTime(&delta_time4, kernel_start4,kernel_stop4);						
						cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu 
						printf("Kernel 3 (mod) is called! \n");
						if(VERBOSE){
							printf("Array Result:\n");
							print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);  
						}
						printf("\n GPU execution with global mem takes: %.3fms",delta_time4);
						printf("--------------------------------------------\n");
                   }break;                                                                   
            default: exit(1); break;                                                                                                         
      }	
	}	
	
	/*Destroy the event created */
	//cudaEventDestroy(kernel_start1);
	cudaEventDestroy(kernel_start2);
	cudaEventDestroy(kernel_start3);
	cudaEventDestroy(kernel_start4);
	//cudaEventDestroy(kernel_stop1);
	cudaEventDestroy(kernel_stop2);
	cudaEventDestroy(kernel_stop3);
	cudaEventDestroy(kernel_stop4);
	
	/*Free the arrays on the CPU*/
	free(cpu_array0);
	free(cpu_array1);
	free(cpu_array_res);
	

    /* Free the arrays on the GPU as now we're done with them */
    cudaFree(gpu_array0);
	cudaFree(gpu_array1);
    cudaFree(gpu_arrayresult);
	
	//Destroy all allocations and reset all state on the current device in the current process
	cudaDeviceReset();
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;
	
	/* code check to make sure blockSize is multiple of WARP */
    if(blockSize<WARP)
	{                        
           blockSize=WARP;
           printf("Warning: Block size specified is less than size of WARP.It got modified to be: %i\n",WARP);     
    }
    else
	{
            if(blockSize % WARP!=0)
            {
                    blockSize=(blockSize+0.5*WARP)/WARP*WARP;
                    printf("Warning: Block size specified is not evenly divisible by the size of WARP.\n");
                    printf("It got modified to be the nearst number that can be evenly divisible by the size of WARP.\n");
                    printf("Now, the blocksize is:%i\n",blockSize);     
            }
    }

	/* code check to make sure Total number of threads is multiple of blockSize*/
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	//execute_gpu_const_test(numBlocks,blockSize);  //  test harness for executing kernel using constant memory
    //execute_gpu_shared_test(numBlocks,blockSize); //  test harness for executing kernel using shared memory 
	execute_gpu_global_test(numBlocks,blockSize); //  test harness for executing kernel using global memory
}
