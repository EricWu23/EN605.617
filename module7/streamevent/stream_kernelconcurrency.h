/*test harness to test kernels that uses global memory and stream*/ 
extern __host__ void execute_gpu_streamkernelconcurrency_arrayAdd(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream);
extern __host__ void execute_gpu_streamkernelconcurrency_arraySubtract(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream);
extern __host__ void execute_gpu_streamkernelconcurrency_arrayMult(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream);
extern __host__ void execute_gpu_streamkernelconcurrency_arrayMod(int numBlocks,int blockSize,int *const gpu_array0,int * const gpu_array1,int* const gpu_arrayresult,int* const h_array_res,cudaStream_t stream);

/* highest level test harness that run all math operations*/
extern __host__ void execute_gpu_streamkernelconcurrency_test(int numBlocks, int blockSize);
extern __host__ void execute_gpu_streamNokernelconcurrency_test(int numBlocks, int blockSize);