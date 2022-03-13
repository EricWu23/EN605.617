/*test harness to test kernels that uses global memory*/ 
extern __host__ void execute_gpu_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/* highest level test harness that run all math operations*/
extern void execute_gpu_global_test(int numBlocks, int blockSize);