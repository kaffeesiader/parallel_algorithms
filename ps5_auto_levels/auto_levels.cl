// #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
// #pragma OPENCL EXTENSION cl_khr_fp64: enable

#define MIN 0
#define MAX 1
#define AVG 2

__kernel void reduce_min_max_avg(__global unsigned char *image, __local unsigned char *scratch,
			 const int length, const int components, __global unsigned char *result)
{
    int gId = get_global_id(0);
    // the start index in global memory
    int gOffset = components*gId;

    int lId = get_local_id(0);
    // the start index in local memory
    int lOffset = 3*components*lId;

    for(int c = 0; c < components; ++c) {
	if(gId < length) {
	    scratch[lOffset + 3*c + MIN] = image[gOffset + c];
	    scratch[lOffset + 3*c + MAX] = image[gOffset + c];
	    scratch[lOffset + 3*c + AVG] = image[gOffset + c];
	} else {
	    scratch[lOffset + 3*c + MIN] = 255;
	    scratch[lOffset + 3*c + MAX] = 0;
	    // neutral element for avg ???
	    scratch[lOffset + 3*c + AVG] = 0;
	}
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // memory organization
    // |R          |G          |B          | ...
    // |min max avg|min max avg|min max avg| ...

    for(int offset = get_local_size(0)/2; offset > 0; offset >>= 1) {
	// check if this work item is responsible for calculation...
	if(lId < offset) {
	    for(int c = 0; c < components; ++c) {
		__local unsigned char *other = &scratch[ lOffset + 3*components*offset + 3*c ];
		__local unsigned char *mine  = &scratch[ lOffset + 3*c ];

		int avg = ((int)mine[AVG] + (int)other[AVG])/2;

		mine[MIN] = (mine[MIN] < other[MIN]) ? mine[MIN] : other[MIN];
		mine[MAX] = (mine[MAX] > other[MAX]) ? mine[MAX] : other[MAX];
		mine[AVG] = (unsigned char)avg;
	    }

	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lId == 0) {

	//  memory organization
	// |Group 0                            |Group 1
	// |R          |G          |B          |R          | ...
	// |min max avg|min max avg|min max avg|min max avg| ...

	for(int c = 0; c < components; ++c) {
	    int rOffset = 3*components*get_group_id(0) + 3*c;

	    result[rOffset + MIN] = scratch[3*c + MIN];
	    result[rOffset + MAX] = scratch[3*c + MAX];
	    result[rOffset + AVG] = scratch[3*c + AVG];
	}

    }

}

__kernel void adjust_image(__global unsigned char *image, __global unsigned char *avg_val,
			   __global float *min_fac, __global float *max_fac, int components)
{
    int gId = get_global_id(0);
    int gOffset = gId * components;

    for(int c = 0; c < components; ++c) {
	int index = gOffset + c;
	unsigned char val = (int)image[index];

	float v = (float)val - (float)avg_val[c];
	v *= (val < avg_val[c]) ? min_fac[c] : max_fac[c];
	v += (float)avg_val[c];

	image[index] = (unsigned char)v;
    }

}
