#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void search(__global VALUE *data, __global int *g_result, VALUE epsilon, VALUE value,
		     int length, __local int *l_result)
{
    int lId = get_local_id(0);
    int l_size = get_local_size(0);
    int lower = lId * l_size;
    int upper = lower + l_size;

    if(upper > length) {
	upper = length;
    }

    l_result[lId] = (int)-1;

    for(int gId = lower; gId < upper; gId++) {
	VALUE mine = data[gId];
	if(fabs(mine - value) < epsilon) {
	    l_result[lId] = get_global_id(0);
	    break;
	}
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(lId == (int)0) {
	g_result[0] = -1;
	for(int i = 0; i < l_size; ++i) {
	    if(l_result[i] >= 0) {
		g_result[0] = l_result[i];
		break;
	    }
	}
    }
}

