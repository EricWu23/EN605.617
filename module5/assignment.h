void print_array(int* arr, int num_row, int num_col);
__device__ void copy_data_to_shared(const int * const data,
									int * const shared_tmp,
									const int num_elements,
									const int tid);
