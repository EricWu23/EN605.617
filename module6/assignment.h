#ifndef __ASSIGNMENT_H__
#define __ASSIGNMENT_H__

/* function to copy data from memory whose starting address specified by pointer shared_tmp 
	to memory whose startign address specified by pointer data.
    globalid is offset for data in unit of element. tid is offset for shared_tmp in unit of element 
*/
__device__ void copy_data_from_shared(int * const data,
									 int * const shared_tmp,
									const int globalid,
									const int tid);
/* function to copy data from memory whose starting address specified by pointer data 
	to memory whose startign address specified by pointer shared_tmp. the number of 
	element to be copyed is specified by num_elements. 
*/									
__device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int num_elements,
									const int tid);
/*function to print out a 2D array for debugging*/
void print_array(int* arr, int num_row, int num_col);
/* function to initialize the data in the array with in-order data*/
void cpu_array0_int(int* arr,int num_row,int num_column);	
/* function to initialize the data in the array with random number between 0 and 3*/
void  cpu_array1_int(int* arr,int num_row,int num_column);

/* kernels that use the global memory*/					
__global__ void arrayAdd(int *array0,int *array1,int* arrayres);
__global__ void arraySubtract(int *array0,int *array1,int* arrayres);
__global__ void arrayMult(int *array0,int *array1,int* arrayres);
__global__ void arrayMod(int *array0,int *array1,int* arrayres);



/*kernels that use the shared memory*/
__global__ void gpu_arrayAdd_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);
__global__ void gpu_arraySubtract_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);
__global__ void gpu_arrayMult_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);
__global__ void gpu_arrayMod_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);

/*kernels that use the constant memory*/ 
__global__ void gpu_arrayAdd_const(int* arrayres,int offset);
__global__ void gpu_arraySubtruct_const(int* arrayres,int offset);
__global__ void gpu_arrayMult_const(int* arrayres,int offset);
__global__ void gpu_arrayMod_const(int* arrayres,int offset);

/* kernels that use the register memory*/					
__global__ void gpu_arrayAdd_register(int *array0,int *array1,int* arrayres);
__global__ void gpu_arraySubtract_register(int *array0,int *array1,int* arrayres);
__global__ void gpu_arrayMult_register(int *array0,int *array1,int* arrayres);
__global__ void gpu_arrayMod_register(int *array0,int *array1,int* arrayres);


/*test harness to test kernels that uses global memory*/ 
__host__ void execute_gpu_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/*test harness to test kernels that use shared memory*/ 
__host__ void execute_gpu_sharedmem_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_sharedmem_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_sharedmem_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_sharedmem_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
/*test harness to test kernels that use constant memory*/ 
__host__ void execute_gpu_constmem_arrayAdd(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_constmem_arraySubtract(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_constmem_arrayMult(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_constmem_arrayMod(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/*test harness to test kernels that uses register memory*/ 
__host__ void execute_gpu_register_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_register_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_register_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_register_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/* highest level test harness that run all math operations*/
void execute_gpu_global_test(int numBlocks, int blockSize);
void execute_gpu_shared_test(int numBlocks, int blockSize);
void execute_gpu_const_test(int numBlocks, int blockSize);
void execute_gpu_register_test(int numBlocks, int blockSize);

#endif /* __ASSIGNMENT_H__ */