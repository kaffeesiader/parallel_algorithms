// #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void matrix_mul(__global VALUE *matrix_U, __global VALUE *matrix_F, __global VALUE *matrix_TMP, VALUE factor)
{
	// two dim parallel
	int i = get_global_id(0)+1;
	int j = get_global_id(1)+1;

	VALUE sum = (VALUE)0.0;
	sum += matrix_U[(i-1)*N + j];
	sum += matrix_U[(i+1)*N + j];
	sum += matrix_U[i*N + j-1];
	sum += matrix_U[i*N + j+1];

	sum -= (factor * matrix_F[i*N + j]);
	sum *= (VALUE)0.25;

	matrix_TMP[N*i + j] = sum;
}

