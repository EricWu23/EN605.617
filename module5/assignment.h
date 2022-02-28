#ifndef __ASSIGNMENT_H__
#define __ASSIGNMENT_H__

void print_array(int* arr, int num_row, int num_col);

__device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int num_elements,
									const int tid);
									
__global__ void arrayAdd(int *array0,int *array1,int* arrayres);

__global__ void gpu_arrayAdd_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);

__global__ void arraySubtract(int *array0,int *array1,int* arrayres);

__global__ void gpu_arraySubtract_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);

__global__ void arrayMult(int *array0,int *array1,int* arrayres);

__global__ void gpu_arrayMult_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);

__global__ void arrayMod(int *array0,int *array1,int* arrayres);

__global__ void gpu_arrayMod_shared(int *array0,int *array1,int* arrayres,int num_elements,int totalnumofThreads);


__host__ void execute_gpu_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_sharedmem_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_sharedmem_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_sharedmem_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

__host__ void execute_gpu_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
__host__ void execute_gpu_sharedmem_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);


void execute_gpu_global_test(int numBlocks, int blockSize);
void execute_gpu_shared_test(int numBlocks, int blockSize);

#endif /* __ASSIGNMENT_H__ */