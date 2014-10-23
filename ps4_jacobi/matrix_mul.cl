#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void matrix_mul(__global VALUE *matrix_A, __global VALUE *matrix_B, __global VALUE *matrix_C)
{
	// sequential
	//for(int i=0; i<N; i++) {
	//	for(int j=0; j<K; j++) {
	//		for(int k=0; k<M; k++) {
	//			matrix_C[N*i+j] += matrix_A[N*i+k] * matrix_B[M*k+j];
	//		}
	//	}
	//}

	// two dim parallel
	int i = get_global_id(0);
	int j = get_global_id(1);
	VALUE sum = 0;
	for(int k=0; k<M; ++k) {
		sum += matrix_A[N*i+k] * matrix_B[M*k+j];
	}
	matrix_C[N*i+j] = sum;
}

