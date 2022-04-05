//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include "assignment.h"
#include "utility.h"
#include "globalmacro.h"
#include "globalpagable.h"
#include "globalpinnedmem.h"
#include "thrustmath.h"

int main(int argc, char** argv)
{
	int totalThreads = (1 << 20);
	int blockSize = 256;
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	int numBlocks = totalThreads/blockSize;
    if(blockSize<WARP1){ 	// code check to make sure blockSize is multiple of WARP1                      
           blockSize=WARP1;
           printf("Warning: Block size specified is less than size of WARP1.It got modified to be: %i\n",WARP1);     
    }
    else{
            if(blockSize % WARP1!=0){
                    blockSize=(blockSize+0.5*WARP1)/WARP1*WARP1;
                    printf("Warning: Block size specified is not evenly divisible by the size of WARP1.\n");
                    printf("It got modified to be the nearst number that can be evenly divisible by the size of WARP1.\n");
					printf("Now, the blocksize is:%i\n",blockSize);     
            }
    }
	if (totalThreads % blockSize != 0) {	// code check to make sure Total number of threads is multiple of blockSize
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	execute_gpu_global_test(numBlocks,blockSize); //  test harness for executing simple math kernel using global memory but pagable host mem
	execute_gpu_pinnedmem_test(numBlocks,blockSize); //  test harness for executing simple math kernel using global memory but pinned Host mem
	execute_gpu_thrust_test(numBlocks,blockSize);// test harness for executing simple math kernel using thrust libary


}
