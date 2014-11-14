// #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void search(__global VALUE *data, volatile __global int *result, VALUE epsilon, VALUE value,
		     int length, __local int *l_result)
{
	int gId = get_global_id(0);
	//int lId = get_local_id(0);
	//int l_size = get_local_size(0);

	// l_result[lId] = (int)-1;

	if(gId < length) {
	    VALUE mine = data[gId];
	    if(((VALUE)fabs(mine - value)) < epsilon) {
		atomic_xchg(result, gId);
	    }
	}

	//barrier(CLK_LOCAL_MEM_FENCE);

	// the first item in each wg needs to check the results of this group
	//if(lId == (int)0) {
	//    for(int i = 0; i < l_size; ++i) {
	//	if(l_result[i] >= 0) {
	//	    // if we have a match, we set the value in global result
	//	    atomic_xchg(result, l_result[i]);
	//	    break;
	//	}
	//    }
	//}
}

