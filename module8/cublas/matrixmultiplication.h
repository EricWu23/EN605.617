
/*
	Description: 
	This function wraps the cublasSgemm from cublas library to perform 2D Matrix multiplication (MatrixA x MatrixB) using GPU 
	and copy result into a memory location on the host pointed by result pointer.
	
	Parameters:
	matrix_A ------pointer to matrixA, matrixA needs to be stored in column-major format
	ht_A     ------Height of matrixA or first dimension of matrixA or the number of rows in matrixA
	wd_A     ------width of matrixA, which is also height of matrixB
	matrix_B ------pointer to matrix_B, matrixB needs to be stored in column-major format
	wd_B     ------width of matrixB or second dimension of matrixB
	result   ------poniter to a memory location that used to store the multiplication result
	stream   ------Specify which stream this function will run 
	Caveat:
		memory location pointed by the result needs to be large enough in size (wd_A*ht_B*sizeof(float)) to store the multiplication result
*/
extern int GPU_MatrixMultiplication(float *matrix_A, int ht_A, int wd_A, float *matrix_B, int wd_B, float *result,cudaStream_t stream);