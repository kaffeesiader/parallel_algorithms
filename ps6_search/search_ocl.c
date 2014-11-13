
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#ifndef WIN32
#include <stdbool.h>
#endif

#include "cl_utils.h"
#include "time_ms.h"
#include "dSFMT.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif

#ifndef LOC_SZ
	#define LOC_SZ 1000
#endif

#define KERNEL_FILE_NAME "./search.cl"
#define VALUE double


cl_uint getDurationMS(cl_event event) {
	cl_ulong start = 0;
	cl_ulong end = 0;
	cl_int ret;

	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), (void *)&start, NULL);
	ret |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void *)&end, NULL);

	CLU_ERRCHECK(ret, "Unable to read profiling information!");

	// the values are measured in nano seconds!
	return (end - start) / 1000000;
}

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

	// allocate memory for data set
	VALUE *data = (VALUE*)malloc(elems*sizeof(VALUE));

	// initialize random number generator
	dsfmt_t rand_state;
	dsfmt_init_gen_rand(&rand_state, (uint32_t)time(NULL));

	// initialize data set (fill randomly)
	for(int j=0; j<elems; ++j) {
		data[j] = dsfmt_genrand_close1_open2(&rand_state);
	}

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);

	// compute local and global size based on maximum group size
	// and number of elements
	int max_group_size = cluGetMaxWorkGroupSize(device_id);
	int local_size = max_group_size;
	int global_size = elems;

	if(elems > max_group_size) {
		global_size = max_group_size;
		while(global_size < elems) {
			global_size += max_group_size;
		}
	} else {
		local_size = elems;
	}

	printf("Input size : %d\n", elems);
	printf("Local size : %d\n", local_size);
	printf("Global size: %d\n", global_size);

	// create memory buffers
	cl_int err;
	cl_mem mem_data = clCreateBuffer(context, CL_MEM_READ_ONLY, elems * sizeof(VALUE), NULL, &err);
	cl_mem mem_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create buffer for matrix");

	// fill memory buffers
	int res = 0;
	err = clEnqueueWriteBuffer(command_queue, mem_data, CL_FALSE, 0, elems * sizeof(VALUE), data, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, mem_result, CL_FALSE, 0, sizeof(int), &res, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DVALUE=%s", EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "search", &err);
	CLU_ERRCHECK(err, "Failed to create 'search' kernel from program");

	VALUE epsilon = 0.4/(VALUE)elems;
	unsigned long long total_time = 0, total_found = 0, found_classical = 0;

	for(int i=0; i<iters; ++i) {
		// search
		unsigned long long start_time = time_ms();
		VALUE val = (VALUE)dsfmt_genrand_close1_open2(&rand_state);
		int result = -1;

		// set arguments and execute kernel
		size_t g_size[] = {elems};
		size_t l_size[] = {LOC_SZ};
		cluSetKernelArguments(kernel, 6,
							  sizeof(cl_mem), (void *)&mem_data,
							  sizeof(cl_mem), (void *)&mem_result,
							  sizeof(VALUE), (void *)&epsilon,
							  sizeof(VALUE), (void *)&val,
							  sizeof(int), (void *)&elems,
							  local_size*sizeof(int), NULL);

		err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, g_size, l_size, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to enqueue 2D kernel");

		// copy results back to host
		err = clEnqueueReadBuffer(command_queue, mem_result, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed reading back result");
		// check the result
		if(result >= 0)
			total_found++;

		total_time += time_ms() - start_time;

		if(find(data, val, epsilon, elems))
			found_classical++;

	}

	// print result
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));	
	printf("Maximum workgroup size: %d\n", cluGetMaxWorkGroupSize((device_id)));
	printf("Done, took %12llu ms, found %12llu\n", total_time, total_found);
	printf("Found with classical method: %llu\n", found_classical);

	free(data);

	// finalization
	err = clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_data);
	err |= clReleaseMemObject(mem_result);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);

	CLU_ERRCHECK(err, "Failed during ocl cleanup");
	
	return EXIT_SUCCESS;
}

