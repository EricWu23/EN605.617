
/*test harness to test kernels that use constant memory*/ 
extern __host__ void execute_gpu_constmem_arrayAdd(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_constmem_arraySubtract(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_constmem_arrayMult(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);
extern __host__ void execute_gpu_constmem_arrayMod(int numBlocks,int blockSize,int *const cpu_array0,int * const cpu_array1,int* const gpu_arrayresult,int* const cpu_array_res);

/* highest level test harness that run all math operations*/
extern void execute_gpu_const_test(int numBlocks, int blockSize);