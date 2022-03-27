/*
	Description: 
		This function wraps around the cusolverDnSSgels from cusolver library to solve linear system equations 
		defined by ht_Axwd_A matrix A and ht_Axnrhs matrix B, and solve for X where A x X= B (matrix multiplication).
	
	Parameters:
		A      ---- the pointer to the matrix A whose dimension is ht_A x wd_A
		ht_A   ---- height of 2D matrix A, which is also the height of matrix B
		wd_A   ---- width of 2D matrix A
		B      ---- the pointer to the matrix B whose dimension is ht_A x nrhs
		nrhs   ---- the width of matrix B
		X      ---- the pointer to the matrix X whose dimension is wd_A x nrhs
		stream ---- Specify which stream this function will run 
*/
void GPU_LinearSolve(float const *A,const int ht_A,const int wd_A, float const *B,const int nrhs,float const *X,cudaStream_t stream);