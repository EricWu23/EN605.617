//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include "assignment.h"
#include "globalmacro.h"

#ifndef HA // this specifies number of rows in matrixA
	#define HA 2
#endif
#ifndef WA // this specifies number of columns in MatrixA
	#define WA 9
#endif
#ifndef WB // this specifies number of columns in MatrixB
	#define WB 2
#endif

#ifndef HA2 // this specifies number of rows in matrixA
	#define HA2 3
#endif
#ifndef WA2 // this specifies number of columns in MatrixA
	#define WA2 3
#endif
#ifndef NRHS // Used by test_GPU_LinearSolve, number of right hand side vectors
	#define NRHS 1
#endif

#include "unitTest_matrixmultiply.h"
#include "unitTest_linearsolve.h"

int main(int argc, char** argv)
{
	test_matrixmultiply(HA,WA,WB);// test harness to test matrix multiplication
	test_GPU_LinearSolve(HA2,WA2,NRHS);//test harness to test linear solver on GPU to solve linear system equations
	if (argc > 1) {
      if (!strcmp(argv[1], "-noprompt") ||!strcmp(argv[1], "-qatest") ){
        return EXIT_SUCCESS;
      }
    } 
    else{
      printf("\nPress ENTER to exit...\n");
      getchar();
    }
	return EXIT_SUCCESS;
}
