#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void search(__global VALUE *data, __global int *result, VALUE epsilon, VALUE value,
		     int length, __local int *l_result)
{
	int lId = get_local_id(0);
	int l_size = get_local_size(0);
	int work_size = length/l_size;

	int lower = lId * work_size;
	int upper = lower + work_size;

	if(upper > length) {
	    upper = length;
	}

	l_result[lId] = (int)-1;

	for(int gId = lower; gId < upper; gId++) {
	    VALUE mine = data[gId];
	    if(fabs(mine - value) < epsilon) {
		l_result[lId] = lId;
		break;
	    }
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lId == (int)0) {
	    result[0] = (int)-1;
	    for(int i = 0; i < l_size; ++i) {
		if(l_result[i] >= 0) {
		    result[0] = l_result[i];
		    break;
		}
	    }
	}
}

