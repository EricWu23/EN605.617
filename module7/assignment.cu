//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include "assignment.h"
#include "utility.h"
#include "globalmacro.h"
#include "globalpagable.h"
#include "globalpinnedmem.h"
#include "sharedmem.h"
#include "constmem.h"
#include "registermem.h"
#include "stream.h"
#include "stream_concurrency.h"
#include "stream_kernelconcurrency.h"


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
    if(blockSize<WARP){ 	// code check to make sure blockSize is multiple of WARP                      
           blockSize=WARP;
           printf("Warning: Block size specified is less than size of WARP.It got modified to be: %i\n",WARP);     
    }
    else{
            if(blockSize % WARP!=0){
                    blockSize=(blockSize+0.5*WARP)/WARP*WARP;
                    printf("Warning: Block size specified is not evenly divisible by the size of WARP.\n");
                    printf("It got modified to be the nearst number that can be evenly divisible by the size of WARP.\n");
					printf("Now, the blocksize is:%i\n",blockSize);     
            }
    }
	if (totalThreads % blockSize != 0) {	// code check to make sure Total number of threads is multiple of blockSize
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	//execute_gpu_global_test(numBlocks,blockSize); //  test harness for executing kernel using global memory but pagable host mem
	execute_gpu_pinnedmem_test(numBlocks,blockSize); //  test harness for executing kernel using global memory but pinned Host mem
	//execute_gpu_shared_test(numBlocks,blockSize); //  test harness for executing kernel using shared memory
	//execute_gpu_const_test(numBlocks,blockSize);  //  test harness for executing kernel using constant memory
	//execute_gpu_register_test(numBlocks,blockSize);  //  test harness for executing kernel using register memory
	execute_gpu_stream_test(numBlocks,blockSize);// test harness for executing kernel using global device mem and stream (all in one stream)
	execute_gpu_streamconcurrency_test(numBlocks,blockSize);//testing harness for the asynchronous memory copy using streams
	execute_gpu_streamkernelconcurrency_test(numBlocks,blockSize);// test harness for testing kernel concurrency using streams
	execute_gpu_streamNokernelconcurrency_test(numBlocks,blockSize);// test harness for testing No kernel concurrency
}
