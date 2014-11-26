#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void scan(__global int *input, __global int *output, int length)
{
	int gId = get_global_id(0);
	int lId = get_local_id(0);
	int l_size = get_local_size(0);
	int pout = 0, pin = 1;

	__local int temp[2*GROUPSIZE];

	if(gId > 0 && gId < length)
	    temp[lId] = input[gId-1];
	else
	    temp[lId] = (int)0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = 1; offset < l_size; offset <<= 1) {
	    pout = 1-pout;
	    pin = 1-pout;

	    if(lId >= offset) {
		int index = (pin * l_size + lId);
		temp[pout*l_size + lId] = temp[index] + temp[index - offset];
	    } else {
		temp[pout*l_size + lId] = temp[pin*l_size + lId];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[gId] = temp[pout*l_size + lId];
}

