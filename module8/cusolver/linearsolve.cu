#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "linearsolve.h"

void GPU_LinearSolve(float const *A,const int ht_A,const int wd_A, float const *B,const int nrhs,float const *X,cudaStream_t stream){
	
	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = ht_A;
	const int n = wd_A;
	int info_gpu = 0;
    int *devInfo = NULL; // info in gpu (device copy)	
	float *d_work = NULL;
	size_t  lwork_bytes = 0;
	int niter=0;
// step 1: create cusolver/cublas handle
	 cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
// step 2: copy A and B to device
	float *d_A,*d_B,*d_X= NULL;
	cudaStat1 = cudaMalloc ((void**)&d_A  ,sizeof(float) * m * n);
	cudaStat2 = cudaMalloc ((void**)&d_B  ,sizeof(float) * m * nrhs);
	cudaStat3 = cudaMalloc ((void**)&d_X  , sizeof(float) * n * nrhs);
	cudaStat4 = cudaMalloc ((void**)&devInfo,sizeof(int));

	assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * m * n , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) * m * nrhs, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
// step 3: query and allocate working space for gels
	cusolverDnSetStream(cusolverH,stream);
	cusolver_status=cusolverDnSSgels_bufferSize(
						cusolverH,
						m,
						n,
						nrhs,
						d_A,
						m,
						d_B,
						m,
						d_X,
						n,
						d_work,
						&lwork_bytes);
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    
	cudaStat1 = cudaMalloc((void**)&d_work,lwork_bytes);
    assert(cudaSuccess == cudaStat1);
// step 4: linear solve
	cusolverDnSetStream(cusolverH,stream);// set the kernel to run on the stream specified by stream
	cusolver_status=cusolverDnSSgels(
						cusolverH,
						m,
						n,
						nrhs,
						d_A,
						m,
						d_B,
						m,
						d_X,
						n,
						d_work,
						lwork_bytes,
						&niter,
						devInfo);
	cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    // check if linear solve is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    printf("\nafter gels: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
	printf("\nafter gels: niter = %d\n", niter);
// step 5: Copy result to Host
    cudaStat1 = cudaMemcpy((void*)X, d_X, sizeof(float)*n*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

// step 6: clean up
	if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
	if (d_X    ) cudaFree(d_X);

    if (cusolverH) cusolverDnDestroy(cusolverH);   

    //cudaDeviceReset();// comment out just to time the code
}
