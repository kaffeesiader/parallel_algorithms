#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct {
    int age;
    char name[32];
} person_t;

__kernel void count(__global person_t *persons, __global int *hist, int length)
{
    int gId = get_global_id(0);
    int lId = get_local_id(0);

    __local int temp[HIST_SIZE];

    if(lId < HIST_SIZE)
	temp[lId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gId < length) {
	int pin = persons[gId].age;
	atomic_inc(&temp[pin]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(lId < HIST_SIZE)
	atomic_add(&hist[lId], temp[lId]);

}

__kernel void scan(__global int *hist)
{
	__local int temp[GROUPSIZE];

	int gId = get_global_id(0);
	int lId = get_local_id(0);

	temp[lId] = gId < HIST_SIZE ? hist[gId] : (int)0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = 1; offset < GROUPSIZE; offset <<= 1) {
	    int mask = (offset << 1) - 1;

	    if((lId & mask) == mask)
		temp[lId] += temp[lId-offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	// set last element to 0
	if(lId == 0)
		temp[GROUPSIZE-1] = (int)0;

	// downsweep phase
	for(int offset = (GROUPSIZE / 2); offset > 0; offset >>= 1) {
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
	if(gId < HIST_SIZE)
	    hist[gId] = temp[lId];
}

__kernel void sort(__global person_t *persons, __global int *hist, __global person_t *output, int length)
{
    int gId = get_global_id(0);

    if(gId < length) {
	int age = persons[gId].age;
	int idx = atomic_inc(&hist[age]);
	output[idx] = persons[gId];
    }
}
