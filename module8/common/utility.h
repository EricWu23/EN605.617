
/* function initializes 2D matrix whose dimension is height x width and is stored in column major format*/
void matrix_init(float *matrix,int height,int width);

/*function to print out a 2D matrix stored in column major format for debugging*/
extern void printMat(float*P,int uWP,int uHP);

/*function to get a time cudaEvent that is created and record on the stream specified by steam*/
extern __host__ cudaEvent_t get_time(unsigned int type,cudaStream_t stream);

