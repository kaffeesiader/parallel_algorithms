
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#include "cl_utils.h"
#include "time_ms.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif
#ifndef LOCAL_SIZE
	#define LOCAL_SIZE 16
#endif

#define KERNEL_FILE_NAME "./downsweep.cl"

void prefix_sum(int *input, int *output, int n) {
	output[0] = 0;
	for(int i = 1; i < n; ++i) {
		output[i] = output[i-1] + input[i-1];
	}
}

void print_array(int *arr, int n) {
	printf("[");
	for (int i = 0; i < n; ++i) {
		printf(" %d", arr[i]);
	}
	printf(" ]\n");
}

int main(int argc, char **argv)
{	
//	if(argc != 2) {
//		printf("Usage: downsweep [elements]\nExample: downsweep 10\n");
//		return EXIT_FAILURE;
//	}

//	int elems = atoi(argv[1]);
	int elems = 8;
	int data[] = {3,1,7,0,4,1,6,3};

	printf("\n");
	printf("Input:  "); print_array(data, elems);

	// allocate memory for data set
//	int *data = (int*)malloc(elems*sizeof(int));
//	// initialize data set (fill randomly)
//	for(int i=0; i<elems; ++i) {
//		data[i] = i;
//	}

	int *result = (int*)malloc(elems*sizeof(int));

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
	
	// create memory buffers
	cl_int err;
	cl_mem mem_input = clCreateBuffer(context, CL_MEM_READ_ONLY, elems * sizeof(int), NULL, &err);
	cl_mem mem_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, elems * sizeof(int), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create memory buffer");

	// fill memory buffers
	err = clEnqueueWriteBuffer(command_queue, mem_input, CL_FALSE, 0, elems * sizeof(int), data, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DGROUPSIZE=%i", elems);
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "scan", &err);
	CLU_ERRCHECK(err, "Failed to create 'scan' kernel from program");

	size_t glb_sz[1] = { elems };
	size_t loc_sz[1] = { LOCAL_SIZE };

	// set arguments and execute kernel
	cluSetKernelArguments(kernel, 3,
				  sizeof(cl_mem), (void *)&mem_input,
				  sizeof(cl_mem), (void *)&mem_output,
				  sizeof(int), (void *)&elems);

	unsigned long long start_time = time_ms();

	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, glb_sz, loc_sz, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to enqueue kernel");

	// read result
	err = clEnqueueReadBuffer(command_queue, mem_output, CL_TRUE, 0, elems*sizeof(int), result, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to read result");

	unsigned long long total_time = time_ms() - start_time;

	/* ---------------------------- result validation -------------------------------- */

	int *seq_result = malloc(elems*sizeof(int));
	prefix_sum(data, seq_result, elems);

	bool valid = true;
	for(int i = 0; i < elems; ++i)
		valid = (result[i] == seq_result[i]) ? valid : false;

	// print result
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));	
	printf("Done, took %12llu ms\n", total_time);
	printf("Validation result: %s\n", valid ? "OK" : "incorrect!");
	printf("Output: "); print_array(result, elems);
	
	// finalization
	err = clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_input);
	err |= clReleaseMemObject(mem_output);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
	
//	free(data);
	free(result);
	free(seq_result);

	return EXIT_SUCCESS;
}

