/*test harness to test kernels that uses global memory and stream*/ 
extern __host__ void execute_gpu_streamconcurrency_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1);
extern __host__ void execute_gpu_streamconcurrency_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1);
extern __host__ void execute_gpu_streamconcurrency_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1);
extern __host__ void execute_gpu_streamconcurrency_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,int* const h_array0,int* const h_array1);

/* highest level test harness that run all math operations*/
extern void execute_gpu_streamconcurrency_test(int numBlocks, int blockSize);