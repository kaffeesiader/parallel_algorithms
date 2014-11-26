#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void scan(__global int *input, __global int *output, int length, __global int *sumBuffer)
{
	__local int temp[GROUPSIZE];

	int gId = get_global_id(0);
	int lId = get_local_id(0);
	int l_size = get_local_size(0);

	temp[lId] = gId < length ? input[gId] : (int)0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = 1; offset < l_size; offset <<= 1) {
	    int mask = (offset << 1) - 1;

	    if((lId & mask) == mask)
		temp[lId] += temp[lId-offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	// set last element to 0
	if(lId == 0) {
	    // store the total sum of current group in sum buffer
	    sumBuffer[get_group_id(0)] = temp[l_size-1];
	    // then set last element to zero
	    temp[l_size-1] = (int)0;
	}

	// downsweep phase
	for(int offset = (l_size / 2); offset > 0; offset >>= 1) {
	    int mask = (offset << 1) - 1;

	    barrier(CLK_LOCAL_MEM_FENCE);

	    if((lId & mask) == mask) {
		int mine = temp[lId];
		int other = temp[lId - offset];
		temp[lId] += other;
		temp[lId - offset] = mine;
	    }
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	output[gId] = temp[lId];
}

__kernel void add(__global int *data, __global int *sumBuffer, int length)
{
    int gId = get_global_id(0);
    int group_id = get_group_id(0);

    if(gId < length)
	data[gId] += sumBuffer[group_id];
}
