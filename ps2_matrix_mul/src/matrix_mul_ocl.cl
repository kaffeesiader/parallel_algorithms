// #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void vec_add(__global double* A, __global double* B, __global double* C, int N, int M)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    double value = 0;
    for(int k = 0; k < M; ++k) {
	double elem_A = A[row * M + k];
	double elem_B = B[k * M + col];
	value += elem_A * elem_B;
    }

    C[row*N+col] = value;
}
