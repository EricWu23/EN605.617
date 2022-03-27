#include "matrixmultiplication.h"
#include <cublas.h>
#include <stdio.h>
int GPU_MatrixMultiplication(float *matrix_A, int ht_A, int wd_A, float *matrix_B, int wd_B, float *result,cudaStream_t stream)
{
	cublasStatus status;
	cublasInit();// step1. Initialize the cublas libary
	    if (matrix_A == 0) {
			fprintf (stderr, "!!!! host memory allocation error (matrix_A)\n");
			return EXIT_FAILURE;
		}
		if (matrix_B == 0) {
			fprintf (stderr, "!!!! host memory allocation error (matrix_B)\n");
			return EXIT_FAILURE;
		}
			if (result == 0) {
			fprintf (stderr, "!!!! host memory allocation error (result)\n");
			return EXIT_FAILURE;
		}
	/*step2  ALLOCATE Memory ON THE DEVICE*/
	float* D_A; float* D_B; float* D_result;
    status=cublasAlloc(ht_A*wd_A,sizeof(float),(void**)&D_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (D_A)\n");
      return EXIT_FAILURE;
    }
	status=cublasAlloc(wd_A*wd_B,sizeof(float),(void**)&D_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (D_B)\n");
      return EXIT_FAILURE;
    }
    status=cublasAlloc(ht_A*wd_B,sizeof(float),(void**)&D_result);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }	
	/*step3  COPY data from Host to Device*/
	status=cublasSetMatrix(ht_A,wd_A,sizeof(float),matrix_A,ht_A,D_A,ht_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory copy from host to device error (D_A)\n");
      return EXIT_FAILURE;
    }
	status=cublasSetMatrix(wd_A,wd_B,sizeof(float),matrix_B,wd_A,D_B,wd_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory copy from host to device error (D_B)\n");
      return EXIT_FAILURE;
    }
	/*step4  Matrix multiplication by calling Kernel cublasSgemm */
	cublasSetKernelStream(stream);// set the kernel to run on the stream specified by stream
	cublasSgemm('n','n',ht_A,wd_B,wd_A,1,D_A,ht_A,D_B,wd_A,0,D_result,ht_A);
    
	status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
    /*step5  Copy result from GPU to Host */
	    cublasGetMatrix(ht_A,wd_B,sizeof(float),D_result,ht_A,result,ht_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (A)\n");
      return EXIT_FAILURE;
    }
	 /*step6 clean up and shutdown */
	status = cublasFree(D_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (D_A)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(D_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (D_B)\n");
      return EXIT_FAILURE;
    }
	status = cublasFree(D_result);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (D_result)\n");
      return EXIT_FAILURE;
    }
	status = cublasShutdown();//shutdown the cublas library
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }
	return EXIT_SUCCESS;
}