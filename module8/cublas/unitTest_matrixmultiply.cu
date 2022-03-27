#include "matrixmultiplication.h"
#include "utility.h"
#include "unitTest_matrixmultiply.h"
#include <stdio.h>

  
int test_matrixmultiply(int hightA,int widthA,int widthB){
	int ht_A=hightA; int wd_A=widthA;
	int ht_B=widthA; int wd_B=widthB;
	int ht_result=hightA; int wd_result=widthB;
    cudaStream_t stream1; 
    cudaStreamCreate(&stream1);
/* host memory allocation*/
    float *matrix_A = (float*)malloc(ht_A*wd_A*sizeof(float));
    float *matrix_B = (float*)malloc(ht_B*wd_B*sizeof(float));
    float *result = (float*)malloc(ht_result*wd_result*sizeof(float));
/* matrix initialization*/
	matrix_init(matrix_A,ht_A,wd_A);
	matrix_init(matrix_B,ht_B,wd_B);
	
/* Matrix multiplication using GPU*/
    cudaEvent_t start_time = get_time(0,stream1);
	int status=GPU_MatrixMultiplication(matrix_A,ht_A,wd_A,matrix_B,wd_B,result,stream1);
	cudaEvent_t end_time = get_time(1,stream1);
    cudaEventSynchronize(end_time);
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time); 
	printf("Matrix multiplication took: %.3f ms \n", delta); 
/* PRINT OUTPUT*/
    printf("\nMatrix A:\n");
    printMat(matrix_A,wd_A,ht_A);
    printf("\nMatrix B:\n");
    printMat(matrix_B,wd_B,ht_B);
	printf("\nMatrix A multipies Matrix B:\n");
    printMat(result,wd_result,ht_result);
/* clearn up and exit*/
	free(matrix_A);  free(matrix_B);  free (result);
	cudaEventDestroy(start_time);
	cudaEventDestroy(end_time);
	cudaStreamDestroy(stream1);//destroy the stream (blocks host until stream is completed)
	cudaDeviceReset();
	return status;
}