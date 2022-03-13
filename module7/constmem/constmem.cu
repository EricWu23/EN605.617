#include <stdio.h>
#include "constmem.h"
#include "utility.h"
#include "globalmacro.h"

/*kernels that use the constant memory*/ 
__global__ void gpu_arrayAdd_const(int* arrayres,int offset);
__global__ void gpu_arraySubtruct_const(int* arrayres,int offset);
__global__ void gpu_arrayMult_const(int* arrayres,int offset);
__global__ void gpu_arrayMod_const(int* arrayres,int offset);



#define MAXARRAYSIZE 8192// this needs to be specify as multiple of blocksize
__constant__  static int constarray0[MAXARRAYSIZE];
__constant__  static int constarray1[MAXARRAYSIZE];
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