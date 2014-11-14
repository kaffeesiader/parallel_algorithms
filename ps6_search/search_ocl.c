
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#include "cl_utils.h"
#include "time_ms.h"
#include "dSFMT.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif
#define VALUE float
#define KERNEL_FILE_NAME "./search.cl"
#define MIN(x,y) ((x<y) ? x : y)

bool find(VALUE *data, VALUE val, VALUE epsilon, int N) {
	for(int j=0; j<N; ++j) {
		if(fabs(data[j] - val) < epsilon) {
			return true;
		}
	}
	return false;
}

int main(int argc, char **argv)
{	
	if(argc != 3) {
		printf("Usage: search [elements] [iterations]\nExample: search 1000000 1000\n");
		return -1;
	}

	int elems = atoi(argv[1]), iters = atoi(argv[2]);

	// initialize random number generator
	dsfmt_t rand_state;
	dsfmt_init_gen_rand(&rand_state, (uint32_t)time(NULL));

	// allocate memory for data set
	VALUE *data = (VALUE*)malloc(elems*sizeof(VALUE));
	// initialize data set (fill randomly)
	for(int j=0; j<elems; ++j) {
		data[j] = dsfmt_genrand_close1_open2(&rand_state);
	}

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
	
	// we use the maximum amount of possible threads
//	int max_group_size = cluGetMaxWorkGroupSize(device_id);
	int max_group_size = 128;
	int global_size = elems;
	int group_size = MIN(max_group_size, elems);
	
	if(elems > group_size) {
		global_size = group_size;
		while(global_size < elems) {
			global_size += group_size;
		}
	}

	printf("Input size    : %d\n", elems);
	printf("Max group size: %d\n", max_group_size);
	printf("Global size   : %d\n", global_size);
	printf("Group size    : %d\n", group_size);
	printf("Local mem size: %lu\n", cluGetLocalMemorySize(device_id));
	
	// create memory buffers
	cl_int err;
	cl_mem mem_data = clCreateBuffer(context, CL_MEM_READ_ONLY, elems * sizeof(VALUE), NULL, &err);
	cl_mem mem_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create buffer for matrix");

	// fill memory buffers
	err = clEnqueueWriteBuffer(command_queue, mem_data, CL_FALSE, 0, elems * sizeof(VALUE), data, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DVALUE=%s", EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "search", &err);
	CLU_ERRCHECK(err, "Failed to create 'search' kernel from program");

	VALUE epsilon = 0.4/(VALUE)elems;
	unsigned long long total_time = 0, total_found = 0;
	unsigned long long found_classical = 0;
	// global and local size are equal...
	size_t glb_sz[1] = { global_size };
	size_t loc_sz[1] = { group_size };

	for(int i = 0; i < iters; ++i) {
		// search
		unsigned long long start_time = time_ms();

		VALUE val = (VALUE)dsfmt_genrand_close1_open2(&rand_state);
		// reset global result value
		int init = -1;
		err = clEnqueueWriteBuffer(command_queue, mem_result, CL_FALSE, 0, sizeof(int), &init, 0, NULL, NULL);
		// set arguments and execute kernel
		cluSetKernelArguments(kernel, 6,
				      sizeof(cl_mem), (void *)&mem_data,
				      sizeof(cl_mem), (void *)&mem_result,
				      sizeof(VALUE), (void *)&epsilon,
				      sizeof(VALUE), (void *)&val,
				      sizeof(int), (void *)&elems,
				      group_size*sizeof(int), NULL);

		err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, glb_sz, loc_sz, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to enqueue kernel");

		// read result
		int result;
		err = clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to read result");
		
		total_time += time_ms() - start_time;
		//printf("Result: %d\n", result);

		if(result > 0)
			total_found++;

		if(find(data, val, epsilon, elems))
			found_classical++;
	}

	// print result
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));	
	printf("Done, took %12llu ms, found %12llu\n", total_time, total_found);
	printf("Found with classical method: %llu\n", found_classical);
	
	// finalization
	err = clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_data);
	err |= clReleaseMemObject(mem_result);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
	
	free(data);

	return EXIT_SUCCESS;
}

