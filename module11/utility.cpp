#include <ctime>
#include <random>
#include "utility.h"
template <class T>
void cpu_array0_int(T* arr,int num_row,int num_column){	
	std::mt19937 rng( std::time(nullptr) ) ;
	std::uniform_int_distribution<T> uniform( 0, 1) ;		
	 for(int i=0; i<num_row; i++){
			for(int j=0; j<num_column; j++){		
				 //arr[i*num_column+j]=i*num_column+j;
				 arr[i*num_column+j]=uniform(rng);// generate value of second array element as a random number between 0 and 1
			}    
	 }				
}
template void cpu_array0_int<unsigned int>(unsigned int* arr,int num_row,int num_column);