#include "utility.h"
#include <stdio.h>
__device__ void copy_data_from_shared(int * const data,
									 int * const shared_tmp,
									const int globalid,
									const int tid)
{
	// deepcopy
	__syncthreads();// synchronize all the threads within a block
	data[globalid]=shared_tmp[tid];
}
__device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int globalid,
									const int tid)
{   // deepcopy
	shared_tmp[tid] = data[globalid];
	__syncthreads();// synchronize all the threads within a block
}

void cpu_array0_int(int* arr,int num_row,int num_column){			
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){		
				 arr[i*num_column+j]=i*num_column+j;
			}    
	 }				
}
void  cpu_array1_int(int* arr,int num_row,int num_column){		
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){
				 //arr[i][j]=i*num_column+j;// the first array contain value from 0 to (totalThreads-1)
				 arr[i*num_column+j]=rand() % 4;// generate value of second array element as a random number between 0 and 3
			}    	 
	 }				
}
void print_array(int* arr, int num_row, int num_col){
	printf("--------------------------------------------\n");
      for(int i=0; i<num_col; i++){
            for(int j=0; j<num_row; j++){
              if (i== num_col-1){
                  printf("%i\n", arr[j*num_col+i]);
              }
              else{
				  printf("%i ", arr[j*num_col+i]);
              }
            }
      }
      printf("--------------------------------------------\n");      
}