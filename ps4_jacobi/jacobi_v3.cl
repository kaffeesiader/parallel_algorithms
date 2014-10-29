#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void jacobi(__global VALUE *matrix_U, __global VALUE *matrix_F, __global VALUE *matrix_TMP,
			 __local VALUE *localData, VALUE factor)
{

	// position in global memory
	int globalRowSize = get_global_size(0);
	int globalMemOffset = get_global_id(0) * globalRowSize + get_global_id(1);

	// position in local memory
	int localRowSize = get_local_size(0) + 2;
	int localMemOffset = (get_local_id(0) + 1) * localRowSize + get_local_id(1) + 1;

	// read the value at given global position into correct location in local memory
	localData[localMemOffset] = matrix_U[globalMemOffset];

	// top border
	if(get_local_id(0) == 0) {
	    if(get_global_id(0) == 0) {
		localData[localMemOffset - localRowSize] = (VALUE)0.0;
	    }
	    else {
		localData[localMemOffset - localRowSize] = matrix_U[globalMemOffset - globalRowSize];
	    }
	}
	// bottom border
	else if(get_local_id(0) == (get_local_size(0) - 1)) {

	    if(get_global_id(0) == (get_global_size(0) - 1)) {
		localData[localMemOffset + localRowSize] = (VALUE)0.0;
	    }
	    else {
		localData[localMemOffset + localRowSize] = matrix_U[globalMemOffset + globalRowSize];
	    }

	}
	// left border
	if(get_local_id(1) == 0) {

	    if(get_global_id(1) == 0) {
		localData[localMemOffset - 1] = (VALUE)0.0;
	    }
	    else {
		localData[localMemOffset - 1] = matrix_U[globalMemOffset - 1];
	    }

	}
	// right border
	else if(get_local_id(1) == (get_local_size(1) - 1)) {

	    if(get_global_id(1) == (get_global_size(1) - 1)) {
		localData[localMemOffset + 1] = (VALUE)0.0;
	    }
	    else {
		localData[localMemOffset + 1] = matrix_U[globalMemOffset + 1];
	    }

	}

	// wait until all data is available in local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute value for given position
	VALUE result = localData[localMemOffset - localRowSize] +
		       localData[localMemOffset + localRowSize] +
		       localData[localMemOffset - 1]   +
		       localData[localMemOffset + 1]   -
		       (factor * matrix_F[globalMemOffset]);

	result *= (VALUE)1/4;

	// write result back into global memory
	matrix_TMP[globalMemOffset] = result;
}

