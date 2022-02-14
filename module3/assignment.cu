//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
    
#ifndef ARRAY_SIZE_X  
    #define ARRAY_SIZE_X 512 // column of the 2D array// this can be defined in Makefile through commandline overide (-D flag for compiler)
#endif
    
#define ARRAY_SIZE_Y 1  //row of the 2D array

#define WARP 32
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(int)))
    
    

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
int cpu_arrayadd [ARRAY_SIZE_Y][ARRAY_SIZE_X];
int cpu_arraysubtract [ARRAY_SIZE_Y][ARRAY_SIZE_X];
int cpu_arraymult [ARRAY_SIZE_Y][ARRAY_SIZE_X];
int cpu_arraymod [ARRAY_SIZE_Y][ARRAY_SIZE_X];

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
  
    for(int kernel=0; kernel<4; kernel++)
    {
      switch(kernel)
      {
            case 0:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arrayAdd<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays
    
                    auto stop = std::chrono::high_resolution_clock::now();
                                
                    cudaMemcpy(cpu_arrayadd, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                    
                     printf("Kernel 0 (Add) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_arrayadd);
                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                    } break;
                        
                                
                                
            case 1:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arraySubtract<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(cpu_arraysubtract, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                     
                        printf("Kernel 1 (subtract) is called! \n");
                         printf("Array Result:\n");
                         print_array(cpu_arraysubtract);
                        std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";  
                         printf("--------------------------------------------\n");
                   }break;                    
                                
                                
           case 2:{
    
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMult<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(cpu_arraymult, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
    
    
                     printf("Kernel 2 (multiplication) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_arraymult);  
                    std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
           case 3:{      
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMod<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
                     cudaMemcpy(cpu_arraymod, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                     printf("Kernel 3 (mod) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_arraymod);   
                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
            default: exit(1); break;                    
                                                             
                                
      }
      
     
                                
                              
    }
                                
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
