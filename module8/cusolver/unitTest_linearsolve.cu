#include "linearsolve.h"
#include "unitTest_linearsolve.h"
#include "utility.h"
#include <stdio.h>

void test_GPU_LinearSolve(const int ht_A,const int wd_A,const int nrhs){
    cudaStream_t stream1; 
    cudaStreamCreate(&stream1);
	/* step1- Host array initialization*/
		float *A=(float*)malloc(ht_A*wd_A*sizeof(float));
		float *B=(float*)malloc(ht_A*nrhs*sizeof(float));
		float *X=(float*)malloc(wd_A*nrhs*sizeof(float));
	if(ht_A ==3 && wd_A==3 && nrhs==1){
		float arrayA[ht_A*wd_A] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0}; 
		float arrayB[ht_A*nrhs] = { 6.0, 15.0, 4.0};
		memcpy(A,arrayA,ht_A*wd_A*sizeof(float));
		memcpy(B,arrayB,ht_A*nrhs*sizeof(float));
		memset(X,0.0, wd_A*nrhs*sizeof(float));		
	}
	else{
		matrix_init(A,ht_A,wd_A);
		matrix_init(B,ht_A,nrhs);
		//initialize memory to zeros
		memset(X,0.0, wd_A*nrhs*sizeof(float));
	}
	printf("\nMatrix A:\n");
    printMat(A,wd_A,ht_A);
    printf("\nMatrix B:\n");
    printMat(B,nrhs,ht_A);
	/* step2- Calling GPU_LinearSolve*/
	cudaEvent_t start_time = get_time(0,stream1);
	GPU_LinearSolve(A,ht_A,wd_A,B,nrhs,X,stream1);
	cudaEvent_t end_time = get_time(1,stream1);
	float delta = 0;
	cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&delta, start_time, end_time); 
	printf("Linear Solver took: %.3f ms \n", delta); 
	/* step3- PRINT OUTPUT */
    printf("\nSolution X:\n");
	printMat(X,nrhs,wd_A);
	/* step4 - clean up*/
	free(A);free(B);free(X);
	cudaEventDestroy(start_time);
	cudaEventDestroy(end_time);
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)
	cudaDeviceReset();
}