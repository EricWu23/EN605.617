#include "utility.h"
#include <stdio.h>
#define column_major_index(i,j,ld) (((j)*(ld))+(i))

void matrix_init(float *matrix,int height,int width){
	 int i,j;
	for (i=0;i<height;i++){
      for (j=0;j<width;j++){
        matrix[column_major_index(i,j,height)] = (float) column_major_index(i,j,height); 
	  }
	}
}

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[column_major_index(i,j,uHP)]);
  }
}

__host__ cudaEvent_t get_time(unsigned int type,cudaStream_t stream)
{
	cudaEvent_t time;
	if(type==0){	
		cudaEventCreate(&time);
	}
	else{
		cudaEventCreateWithFlags(&time,cudaEventBlockingSync);
		
	}
	cudaEventRecord(time,stream);
	return time;
}