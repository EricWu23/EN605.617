#include <stdio.h>

//From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;//global index calculation
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<8;
  float *x, *y, *d_x, *d_y;
  // (pagable) host mem allocation
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
// device memory allocation
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
// host memory initialization
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
// explicity memory copy: 
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);// x-->d_x
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);//y-->d_y

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
// explicity memory copy: 
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);//d_y--->y

  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(y[i]-4.0f));
    printf("y[%d]=%f\n",i,y[i]);
  }
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
