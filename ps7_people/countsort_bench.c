#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "people.h"
#include "cl_utils.h"
#include "time_ms.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif
#ifndef LOCAL_SIZE
	#define LOCAL_SIZE 128
#endif

#define KERNEL_FILE_NAME "./countsort.cl"
#define HIST_SIZE (MAX_AGE+1)

void exclusive_prefix_sum(int *input, int n) {
	int sum = 0, tmp = 0;
	for(int i = 0; i < n; ++i) {
		tmp = input[i];
		input[i] = sum;
		sum += tmp;
	}
}

void count_sort(person_t *persons, person_t *sorted, int n) {

	int C[MAX_AGE];
	memset(C, 0, MAX_AGE*sizeof(int));
	// compute histogram
	for (int i = 0; i < n; ++i)
		C[persons[i].age]++;

	exclusive_prefix_sum(C, MAX_AGE);

	for (int i = 0; i < n; ++i) {
		int value = persons[i].age;
		int idx = C[value]++;
		sorted[idx] = persons[i];
	}
}

void print_array(int *arr, int n) {
	printf("[");
	for (int i = 0; i < n; ++i) {
		printf(" %d", arr[i]);
	}
	printf(" ]\n");
}

bool check_sorted(person_t *list, int n) {
	if(n > 1) {
		for(int i = 0; i < n-1; ++i) {
			if(list[i+1].age < list[i].age)
				return false;
		}
	}
	return true;
}

int main(int argc, char **argv)
{

	if(argc != 3) {
		printf("Usage: hillissteele [elements] [seed]\nExample: hillissteele 10 123\n");
		return EXIT_FAILURE;
	}

	int elems = atoi(argv[1]); unsigned int seed = atoi(argv[2]);
	srand(seed);

	person_t *persons = malloc(elems*sizeof(person_t));
	person_t *output = malloc(elems*sizeof(person_t));
	fill_list(persons, elems);

//	print_persons(persons, elems);

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);

	// create memory buffers
	cl_int err;
	cl_mem mem_input = clCreateBuffer(context, CL_MEM_READ_ONLY, elems * sizeof(person_t), NULL, &err);
	cl_mem mem_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, elems * sizeof(person_t), NULL, &err);
	cl_mem mem_hist = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE * sizeof(int), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create memory buffer");

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DHIST_SIZE=%i -DGROUPSIZE=%i", HIST_SIZE, LOCAL_SIZE);
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel count_kernel = clCreateKernel(program, "count", &err);
	CLU_ERRCHECK(err, "Failed to create 'count' kernel from program");
	cl_kernel scan_kernel = clCreateKernel(program, "scan", &err);
	CLU_ERRCHECK(err, "Failed to create 'scan' kernel from program");
	cl_kernel sort_kernel = clCreateKernel(program, "sort", &err);
	CLU_ERRCHECK(err, "Failed to create 'sort' kernel from program");

	unsigned long long start_time = time_ms();

	// fill memory buffers
	err = clEnqueueWriteBuffer(command_queue, mem_input, CL_FALSE, 0, elems * sizeof(person_t), persons, 0, NULL, NULL);

	int zero = 0;
	err |= clEnqueueFillBuffer(command_queue, mem_hist, (void *)&zero, sizeof(int), 0, HIST_SIZE * sizeof(int), 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	/* --------------------------- Calculate histogram -------------------------------- */

	// ensure that global size is at least local size
	// and is devideable by local size...
	int global_size = elems;
	if((global_size % LOCAL_SIZE) > 0) {
		global_size += (LOCAL_SIZE - (global_size % LOCAL_SIZE));
	}

	size_t glb_sz[] = { global_size };
	size_t loc_sz[] = { LOCAL_SIZE };

	// set arguments and execute kernel
	cluSetKernelArguments(count_kernel, 3,
				  sizeof(cl_mem), (void *)&mem_input,
				  sizeof(cl_mem), (void *)&mem_hist,
				  sizeof(int), (void *)&elems);

	err = clEnqueueNDRangeKernel(command_queue, count_kernel, 1, NULL, glb_sz, loc_sz, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to enqueue 'count' kernel");

	/* --------------------------- Calculate prefix sum ------------------------------ */

	glb_sz[0] = LOCAL_SIZE;

	// set arguments and execute kernel
	cluSetKernelArguments(scan_kernel, 1,
				  sizeof(cl_mem), (void *)&mem_hist);

	err = clEnqueueNDRangeKernel(command_queue, scan_kernel, 1, NULL, glb_sz, loc_sz, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to enqueue 'scan' kernel");

	// read result
//	err = clEnqueueReadBuffer(command_queue, mem_hist, CL_TRUE, 0, HIST_SIZE*sizeof(int), hist, 0, NULL, NULL);
//	CLU_ERRCHECK(err, "Failed to read result");

	/* ------------------------------- Do the sorting -------------------------------- */

	glb_sz[0] = global_size;

	// set arguments and execute kernel
	cluSetKernelArguments(sort_kernel, 4,
				  sizeof(cl_mem), (void *)&mem_input,
				  sizeof(cl_mem), (void *)&mem_hist,
				  sizeof(cl_mem), (void *)&mem_output,
				  sizeof(int), (void *)&elems);

	err = clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, glb_sz, loc_sz, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to enqueue 'sort' kernel");

	// read result
	err = clEnqueueReadBuffer(command_queue, mem_output, CL_TRUE, 0, elems*sizeof(person_t), output, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to read result");

	unsigned long long ocl_time = time_ms() - start_time;

	/* ---------------------------- sequential sorting ------------------------------- */

	start_time = time_ms();
	count_sort(persons, output, elems);
	unsigned long long seq_time = time_ms() - start_time;

	/* ---------------------------- result validation -------------------------------- */

	bool valid = check_sorted(output, elems);
	// print result
//	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
	printf("Sequential  : %12llu ms\n", seq_time);
	printf("OCL Device %d: %12llu ms\n", CL_DEVICE, ocl_time);
	printf("Validation result: %s\n", valid ? "OK" : "incorrect!");

	// finalization
	err = clFinish(command_queue);
	err |= clReleaseKernel(count_kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(mem_input);
	err |= clReleaseMemObject(mem_output);
	err |= clReleaseMemObject(mem_hist);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	err |= clReleaseDevice(device_id);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");

	free(persons);
	free(output);

	return EXIT_SUCCESS;
}

