//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
    
#define WARP 32

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
void print_array(int** arr, int num_row, int num_col)
{
    
          for(int i=0; i<num_col; i++)
      {
            for(int j=0; j<num_row; j++)
            {
              if (i== num_col-1)
                {
					//arr[0][0]=0;
                  printf("%i\n", arr[j][i]);
                }
              else
                {
					//arr[0][0]=0;
					
                  //printf("%i ", arr[j][i]);
				  //printf("%i ", *(*(arr+j)+i));
				  //*(*(arr+j))=0;
				  printf("%i ", *(*(arr+j)));
				  
                }
    
            }
         
     }
    
    
    
}


/* Dynamically allocate an 2D array using pagable memory on the host and return the pointer */
int** cpu_2darray_Malloc(int num_row,int num_column)
{

	 // allocate a 2D array dynamically (pagable memory)
	 int** cpu_arr = (int**)malloc(num_row * sizeof(int*));
	 for (int i = 0; i < num_row; i++)
	 {
			cpu_arr[i] = (int*)malloc(num_column * sizeof(int));
	 }

	 return cpu_arr;
}

/* Dynamically allocate an 2D array using pinned memory on the host and return the pointer */
void cpu_2darray_cudaMallocHost(int** cpu_arr,size_t size)
{
 

}



/* initialize the data in the array according to assignment requirement*/
void cpu_array0_int(int** arr,int num_row,int num_column){
		
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++)
	 {
			for(int j=0; j<num_column; j++)
			{
				 arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
			}    
	 
	 }				
}
/* initialize the data in the array according to assignment requirement*/
void  cpu_array1_int(int** arr,int num_row,int num_column){
		
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++)
	 {
			for(int j=0; j<num_column; j++)
			{
				 //arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
				 arr[i][j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
			}    
	 
	 }		
		
}

void main_sub0(int numBlocks,int blockSize)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;
	int cpu_arr_size_x=totalThreads;
    int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
	
	/* dynamically allocate the pinned memory on the host*/
	int **cpu_array0,**cpu_array1,**cpu_array_res; 

    // cpu_array0 = (int **)malloc(sizeof(int *) * cpu_arr_size_y);
    // cpu_array0[0] = (int *)malloc(sizeof(int) * cpu_arr_size_x* cpu_arr_size_y);	
	// for(int i = 1; i < cpu_arr_size_y; i++){
		 // cpu_array0[i] = cpu_array0[0] + i * cpu_arr_size_x;
	// }
			
	cudaMallocHost((void**)&cpu_array0,sizeof(int *) * cpu_arr_size_y);	
	cudaMallocHost((void**)&(*cpu_array0),sizeof(int) * cpu_arr_size_x* cpu_arr_size_y);	
	for(int i = 1; i < cpu_arr_size_y; i++){
			printf("I should not be here! \n");
		 cpu_array0[i] = *cpu_array0 + i * cpu_arr_size_x;
	} 	
	
	printf("I am here 1! \n");
	cpu_array1 = (int **)malloc(sizeof(int *) * cpu_arr_size_y);
    cpu_array1[0] = (int *)malloc(sizeof(int) * cpu_arr_size_x* cpu_arr_size_y);
	for(int i = 1; i < cpu_arr_size_y; i++){
		cpu_array1[i] = cpu_array1[0] + i * cpu_arr_size_y;
	}
	
	cpu_array_res = (int **)malloc(sizeof(int *) * cpu_arr_size_y);
    cpu_array_res[0] = (int *)malloc(sizeof(int) * cpu_arr_size_x* cpu_arr_size_y);
		for(int i = 1; i < cpu_arr_size_y; i++){
		cpu_array_res[i] = cpu_array_res[0] + i * cpu_arr_size_y;
	}
	
	printf("I am here 2! \n");
	
    /* data init*/
    //cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	//printf("I am here 3! \n");
	//cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	
	
	/* print out the arrays for debuging */
	printf("The following two arrays are initialized on cpu! \n");
	printf("Array0:\n");
	print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	printf("Array1:\n");
	print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);	
     
    /* layout specification
     1. assume that blockSize is at least 64 and will be multiple of 32
     2. numberBlocks will be at least 1
    */
    const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
    
    /* Declare statically arrays */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
    

    //printf("size_in_bytes:%i\n",size_in_bytes);
    // memory allocation on GPU
    cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
    
    // memory copy from cpu to gpu
    cudaMemcpy( gpu_array0,*cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,*cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
  
    for(int kernel=0; kernel<4; kernel++)
    {
      switch(kernel)
      {
            case 0:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arrayAdd<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays
    
                    auto stop = std::chrono::high_resolution_clock::now();
                                
                    cudaMemcpy(*cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                    
                     printf("Kernel 0 (Add) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);
                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                    } break;
                        
                                
                                
            case 1:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arraySubtract<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(*cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                     
                        printf("Kernel 1 (subtract) is called! \n");
                         printf("Array Result:\n");
                         print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);
                        std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";  
                         printf("--------------------------------------------\n");
                   }break;                    
                                
                                
           case 2:{
    
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMult<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(*cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
    
    
                     printf("Kernel 2 (multiplication) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); 
                    std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
           case 3:{      
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMod<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
                     cudaMemcpy(*cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                     printf("Kernel 3 (mod) is called! \n");
                     printf("Array Result:\n");
                     print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);  
                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
            default: exit(1); break;                    
                                                             
                                
      }
      
     
                                
                              
    }
    /*Free the arrays on the CPU*/
	//free(*cpu_array0);
	//free(*cpu_array1);
	//free(*cpu_array_res);
    //free((void *)cpu_array0[0]);
	//free((void *)cpu_array0);
	
	// for(int i = 0; i < cpu_arr_size_y; i++)
	// {
		// free((void *)cpu_array0[i]);
	// }
	// free((void *)cpu_array0);

	
	free((void *)cpu_array1[0]);
	free((void *)cpu_array1);
	
	free((void *)cpu_array_res[0]);
	free((void *)cpu_array_res);
	
	cudaFreeHost((void *)cpu_array0[0]);
	cudaFreeHost((void *)cpu_array0);
	//cudaFreeHost(*cpu_array0);
    /* Free the arrays on the GPU as now we're done with them */
    cudaFree(gpu_array0);
	cudaFree(gpu_array1);
    cudaFree(gpu_arrayresult);

}




int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 10);
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
