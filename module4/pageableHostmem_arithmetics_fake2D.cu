//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
    
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
    


// function to print out a 2D array for debugging
void print_array(int* arr, int num_row, int num_col)
{
      printf("--------------------------------------------\n");
          for(int i=0; i<num_col; i++)
      {
            for(int j=0; j<num_row; j++)
            {
              if (i== num_col-1)
                {
                  printf("%i\n", arr[j*num_col+i]);
                }
              else
                {
	
				  printf("%i ", arr[j*num_col+i]);
				  
                }
    
            }
         
     }
      printf("--------------------------------------------\n");      
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
void cpu_array0_int(int* arr,int num_row,int num_column){
		
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++)
	 {
			for(int j=0; j<num_column; j++)
			{		
				 arr[i*num_column+j]=i*num_column+j;
			}    
	 
	 }				
}
/* initialize the data in the array according to assignment requirement*/
void  cpu_array1_int(int* arr,int num_row,int num_column){
		
	 //2D array intialization 		
	 for(int i=0; i<num_row; i++)
	 {
			for(int j=0; j<num_column; j++)
			{
				 //arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
				 arr[i*num_column+j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
			}    
	 
	 }		
		
}

/* simple Caesar cypher*/
void encrypt(int* arr,int total_element){
                                       
     for(int i=0;i<total_element;i++)
    {
    
        arr[i]=arr[i]+OFFSET;
    
    }
                                                                                                           
}

/* simple Caesar cypher*/
void decrypt(int* arr,int total_element){
                                       
     for(int i=0;i<total_element;i++)
    {
    
        arr[i]=arr[i]-OFFSET;
    
    }
                                                                                                           
}

bool validtest( int* const uut,int total_element,size_t bytes){
                                     
     int* dest;
                                      
     dest=(int *) malloc(bytes);                                 
     memcpy(dest,uut,bytes); //deep copy to create the ground truth
                                 
     encrypt(uut,total_element);
     decrypt(uut,total_element);
     for(int i=0;i<total_element;i++){
             if(uut[i]!=dest[i])
            {
                free(dest);
                return false;
            }   
    }        
    
    free(dest);
    return true;                                  
}                                      
                                       
void main_sub0(int numBlocks,int blockSize)
{
	int totalThreads=numBlocks*blockSize;
	int cpu_arr_size_y=1;//row
	int cpu_arr_size_x=totalThreads;//column
    int size_in_bytes = cpu_arr_size_x* cpu_arr_size_y* sizeof(int);
    int size_in_elements = cpu_arr_size_x* cpu_arr_size_y;
	
	/* dynamically allocate the memory on the host*/
	int *cpu_array0,*cpu_array1,*cpu_array_res; 
    
    auto start1 = std::chrono::high_resolution_clock::now();
	cpu_array0 = (int *) malloc(size_in_bytes);//pagable
	cpu_array1 = (int *)malloc(size_in_bytes);
	cpu_array_res = (int *)malloc(size_in_bytes);
	 auto stop1 = std::chrono::high_resolution_clock::now();
	 std::cout << "Time taken by memory allocation on host: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop1-start1).count() << "ns\n";
    /* data init*/
    cpu_array0_int(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
	cpu_array1_int(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	
	/* print out the arrays for debuging */
	if(VERBOSE)
	{
		printf("The following two arrays are initialized on cpu! \n");
		printf("Array0:\n");
		print_array(cpu_array0,cpu_arr_size_y,cpu_arr_size_x);
		printf("Array1:\n");
		print_array(cpu_array1,cpu_arr_size_y,cpu_arr_size_x);
	}
	
     
    /* test the Caesar cypher*/
	if(VERBOSE){printf("%s\n", validtest(cpu_array0,size_in_elements,size_in_bytes)? "Caesar cypher works!" : "Caesar cypher not working!");}
    
    /* layout specification
         1. assume that blockSize is at least 64 and will be multiple of 32
         2. numberBlocks will be at least 1
    */
    const dim3 threads_layout(WARP,blockSize/WARP); // there are multiple ways of layout to achieve blocksize. I choose to fix the  blockDim.x as the WARP size
    const dim3 blocks_layout(1,numBlocks);// there are multiple ways of layout to achieve numBlocks, I choose to fix the gridDim.x to 1
    
    /* Declare statically arrays */
    int * gpu_array0, * gpu_array1,*gpu_arrayresult;
    

    // memory allocation on GPU
     auto start2 = std::chrono::high_resolution_clock::now();
    cudaMalloc((void **)&gpu_array0, size_in_bytes);
	cudaMalloc((void **)&gpu_array1, size_in_bytes);
    cudaMalloc((void **)&gpu_arrayresult, size_in_bytes);
    auto stop2 = std::chrono::high_resolution_clock::now();
     std::cout << "Time taken by memory allocation on GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop2-start2).count() << "ns\n";
     
    // memory copy from cpu to gpu
     auto start3 = std::chrono::high_resolution_clock::now();
    cudaMemcpy( gpu_array0,cpu_array0 , size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_array1,cpu_array1 , size_in_bytes, cudaMemcpyHostToDevice );
     auto stop3 = std::chrono::high_resolution_clock::now();
      std::cout << "Time taken by memory copy from cpu to gpu: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop3-start3).count() << "ns\n";
  
    for(int kernel=0; kernel<4; kernel++)
    {
      switch(kernel)
      {
            case 0:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arrayAdd<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult); // kernel call to add two 2-D arrays
    
                    auto stop = std::chrono::high_resolution_clock::now();
                                
                    cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                    
                     printf("Kernel 0 (Add) is called! \n");
					 if(VERBOSE)
					 {
					    printf("Array Result:\n");
						print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 
					 }

                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                    } break;
                        
                                
                                
            case 1:{
                    auto start = std::chrono::high_resolution_clock::now();   
                    arraySubtract<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to subtract two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
                     
                        printf("Kernel 1 (subtract) is called! \n");
						 if(VERBOSE){
						    printf("Array Result:\n");
							print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);					 
						 }
                        std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";  
                         printf("--------------------------------------------\n");
                   }break;                    
                                
                                
           case 2:{
    
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMult<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise)multiply two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
    
                     cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu
    
    
                     printf("Kernel 2 (multiplication) is called! \n");
					 if(VERBOSE){
							 printf("Array Result:\n");
							 print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x); 
					 }
                    std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
           case 3:{      
                    auto start = std::chrono::high_resolution_clock::now(); 
                     arrayMod<<<blocks_layout,threads_layout>>>(gpu_array0,gpu_array1,gpu_arrayresult);//kernel call to (elementwise) mod divide two 2-D arrays 
                    auto stop = std::chrono::high_resolution_clock::now();
                     cudaMemcpy(cpu_array_res, gpu_arrayresult, size_in_bytes, cudaMemcpyDeviceToHost); // memcopy from gpu to cpu 
                     printf("Kernel 3 (mod) is called! \n");
                      if(VERBOSE){
						 printf("Array Result:\n");
						 print_array(cpu_array_res,cpu_arr_size_y,cpu_arr_size_x);  
						}
                     std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << "ns\n";
                     printf("--------------------------------------------\n");
                   }break;                    
                                               
            default: exit(1); break;                    
                                                             
                                
      }
      
     
                                
                              
    }
    /*Free the arrays on the CPU*/
	free(cpu_array0);
	free(cpu_array1);
	free(cpu_array_res);
	
	// cudaFreeHost((void *)cpu_array0[0]);
	// cudaFreeHost((void *)cpu_array0);
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
