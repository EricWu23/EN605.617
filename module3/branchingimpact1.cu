/* 
	File Description: 
	This file explores the impact of warp divergence based on simple operations (add, subtract)
	The plan is 
	1. we have two 1x32 array on the host
	2. copy the array from the host memory into GPU memory 
	3. Create three different kernels,
		3.1 kernel1: if the global index is even, perform add, if the index is odd, perform subtract
		3.2 kernel2: if the global index is larger than 15, subtract. If the global index is less or equal to 15, add
		3.3 kernel3: if the global index is less than 10, add. if the global index is between 10 and 25, subtract. if the global index is larger than 25,add  
    4. Time the operation of all the three kernels and see if there is any difference.
	
	The intension is that in all three cases, we are executing 16 adds and 16 subtracts of two ints. However, due to the different way we branch the operation, we expect to 
	obtain different execution speed.
	
	Author: Yujiang Wu
	Date: 2022/02/14

*/ 
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

#ifndef ARRAY_SIZE_X  
    #define ARRAY_SIZE_X 32 // column of the 2D array// this can be defined in Makefile through commandline overide (-D flag for compiler)
#endif
    
#define ARRAY_SIZE_Y 1  //row of the 2D array

#define WARP 32
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(int)))
    
__global__ void Kernel1(int *array0,int *array1,int* arrayresult) {
	  
       // collapse the higher dimension layout or nested layout down to flat 2D
		const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	   // collapse flat 2D down to 1D, whose index is global thread index
		const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
	if(global_idx%2==0)
	{
		arrayresult[global_idx]=array0[global_idx]+array1[global_idx];
	}	
	else{
		arrayresult[global_idx]=array0[global_idx]-array1[global_idx];	
	}                        
}    
__global__ void Kernel2(int *array0,int *array1,int* arrayresult) {
	  
       // collapse the higher dimension layout or nested layout down to flat 2D
		const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	   // collapse flat 2D down to 1D, whose index is global thread index
		const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
	if(global_idx>15)
	{
		arrayresult[global_idx]=array0[global_idx]+array1[global_idx];
	}	
	else
	{
		arrayresult[global_idx]=array0[global_idx]-array1[global_idx];	
	}                        
}    

__global__ void Kernel3(int *array0,int *array1,int* arrayresult) {
	  
       // collapse the higher dimension layout or nested layout down to flat 2D
		const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		const int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	   // collapse flat 2D down to 1D, whose index is global thread index
		const int global_idx = ((gridDim.x * blockDim.x) * idy) + idx;
	if(global_idx<10)
	{
		arrayresult[global_idx]=array0[global_idx]+array1[global_idx];
	}	
	else if(global_idx>=10 && global_idx<=25)
	{
		arrayresult[global_idx]=array0[global_idx]-array1[global_idx];	
	}
	else{
		arrayresult[global_idx]=array0[global_idx]+array1[global_idx];
	}
}    
    
// function to print out a 2D array for debugging
 void print_array(int arr[ARRAY_SIZE_Y][ARRAY_SIZE_X] )
{   
      for(int i=0; i<ARRAY_SIZE_X; i++)
      {
            for(int j=0; j<ARRAY_SIZE_Y; j++)
            {
              if (i== ARRAY_SIZE_X-1)
                {
                  printf("%i\n", arr[j][i]);
                }
              else
                {
                  printf("%i ", arr[j][i]);
                }
    
            }
         
     }
}


/* Declare  arrays on the cpu */

int cpu_array0 [ARRAY_SIZE_Y][ARRAY_SIZE_X];
int cpu_array1 [ARRAY_SIZE_Y][ARRAY_SIZE_X];
int cpu_result [ARRAY_SIZE_Y][ARRAY_SIZE_X];

/* initialize the data in the array according to assignment requirement*/
void cpu_array_int(int numBlocks,int blockSize){
 int totalThreads=numBlocks*blockSize;
    if(totalThreads!=ARRAY_SIZE_X*ARRAY_SIZE_Y)
    {
       printf("Total number of Threads specified from command line does not match total number of data elements in the array. Initialization failed\n");
       printf("The total number of elements in array is :%i\n",ARRAY_SIZE_X*ARRAY_SIZE_Y);
       printf("Either give a commandline argument that match the array size or recompile by modifying the macro defintion ARRAY_SIZE_X");
       exit(1);
    }
    else
    {
         for(int i=0; i<ARRAY_SIZE_Y; i++)
         {
                for(int j=0; j<ARRAY_SIZE_X; j++)
                {
                     cpu_array0[i][j]=i*ARRAY_SIZE_X+j;// the first array contain value from 0 to (totalThreads-1)
                     cpu_array1[i][j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
                }    
         
         }
         printf("The following two arrays are initialized! \n");
         printf("Array0:\n");
         print_array(cpu_array0);
         printf("--------------------------------------------\n");

         printf("Array1:\n");
         print_array(cpu_array1);
        printf("--------------------------------------------\n");
    
    }
     
}

void main_sub0(int numBlocks,int blockSize)
{
    /* data init*/
    cpu_array_int(numBlocks,blockSize);
     
    /* layout specification
     1. assume that blockSize is at least 64 and will be multiple of 32
     2. numberBlocks will be at least 1
    */
    const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
    
    /* Declare statically arrays */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
    
    int size_in_bytes = ARRAY_SIZE_X* ARRAY_SIZE_Y* sizeof(int);
    
    // memory allocation on GPU
    cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
    
    // memory copy from cpu to gpu
    cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
                                             
  auto start = std::chrono::high_resolution_clock::now();
  for(int k=0;k<1000;k++)
  { 
	Kernel1<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
  }   
  auto stop = std::chrono::high_resolution_clock::now();
	
  cudaMemcpy(cpu_result, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
 printf("Kernel1 is called 1000 times! \n");
			printf("Array Result:\n");
			print_array(cpu_result);
			std::cout << "Total Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
			printf("--------------------------------------------\n");
	 
    start = std::chrono::high_resolution_clock::now();
    for(int k=0;k<1000;k++)
    {
        Kernel2<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	}
    stop = std::chrono::high_resolution_clock::now();
	
    cudaMemcpy(cpu_result, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel2 is called 1000 times! \n");
			printf("Array Result:\n");
			print_array(cpu_result);
			std::cout << "Total Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
			printf("--------------------------------------------\n");

    start = std::chrono::high_resolution_clock::now();
    for(int k=0;k<1000;k++)
    {
        Kernel3<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);
	}
    stop = std::chrono::high_resolution_clock::now();
	
    cudaMemcpy(cpu_result, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
	printf("Kernel3 is called 1000 times! \n");
			printf("Array Result:\n");
			print_array(cpu_result);
			std::cout << "Total Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
			printf("--------------------------------------------\n");

   
    
    /* Free the arrays on the GPU as now we're done with them */
    cudaFree(gpu_array0);
	cudaFree(gpu_array1);
    cudaFree(gpu_arrayresult);

}




int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
        printf("Total number of threads changed to:%i\n", totalThreads);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
        printf("number of thread per block changed to:%i\n", blockSize);
	}

	int numBlocks = totalThreads/blockSize;
    
    
    /* code check to make sure blockSize is multiple of WARP */
    if(blockSize<WARP){                        
           blockSize=WARP;
           printf("Warning: Block size specified is less than size of WARP.It got modified to be: %i\n",WARP);     
         }
     else{
            if(blockSize % WARP!=0)
            {
                    blockSize=(blockSize+0.5*WARP)/WARP*WARP;
                    printf("Warning: Block size specified is not evenly divisible by the size of WARP.\n");
                    //printf("It got modified to be the nearst number that can be evenly divisible by the size of WARP.\n");
                    // printf("Now, the blocksize is:%i\n",blockSize);     
            }
         }
    
	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    
    main_sub0(numBlocks,blockSize);

}
