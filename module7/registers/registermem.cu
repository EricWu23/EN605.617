#include <stdio.h>
#include "registermem.h"
#include "utility.h"
#include "globalmacro.h"

/* kernels that use the register memory*/					
static __global__ void gpu_arrayAdd_register(int *array0,int *array1,int* arrayres);
static __global__ void gpu_arraySubtract_register(int *array0,int *array1,int* arrayres);
static __global__ void gpu_arrayMult_register(int *array0,int *array1,int* arrayres);
static __global__ void gpu_arrayMod_register(int *array0,int *array1,int* arrayres);

#define NUM_REGISTER_PER_THREAD 256
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
