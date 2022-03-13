/* function to copy data from memory whose starting address specified by pointer shared_tmp 
	to memory whose startign address specified by pointer data.
    globalid is offset for data in unit of element. tid is offset for shared_tmp in unit of element 
*/
extern __device__ void copy_data_from_shared(int * const data,
									 int * const shared_tmp,
									const int globalid,
									const int tid);
/* function to copy data from memory whose starting address specified by pointer data 
	to memory whose startign address specified by pointer shared_tmp. the number of 
	element to be copyed is specified by num_elements. 
*/									
extern __device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int num_elements,
									const int tid);
/*function to print out a 2D array for debugging*/
extern void print_array(int* arr, int num_row, int num_col);
/* function to initialize the data in the array with in-order data*/
extern void cpu_array0_int(int* arr,int num_row,int num_column);	
/* function to initialize the data in the array with random number between 0 and 3*/
extern void  cpu_array1_int(int* arr,int num_row,int num_column);