/*test harness to test kernels that use shared memory*/ 
extern __host__ void execute_gpu_sharedmem_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_sharedmem_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_sharedmem_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_sharedmem_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/* highest level test harness that run all math operations*/
extern void execute_gpu_shared_test(int numBlocks, int blockSize);