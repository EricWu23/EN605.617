#include "utility.h"
template <class T>
void cpu_array0_int(T* arr,int num_row,int num_column){			
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){		
				 arr[i*num_column+j]=i*num_column+j;
			}    
	 }				
}
template void cpu_array0_int<unsigned int>(unsigned int* arr,int num_row,int num_column);