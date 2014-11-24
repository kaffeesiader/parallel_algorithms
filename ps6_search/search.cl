#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void search(__global VALUE *data, volatile __global int *result, VALUE epsilon, VALUE value,
		     int length)
{
	int gId = get_global_id(0);
	
	if(gId < length) {
	    VALUE mine = data[gId];
	    if(((VALUE)fabs(mine - value)) < epsilon) {
		atomic_xchg(result, gId);
		//result[0] = gId;
	    }
	}

}

