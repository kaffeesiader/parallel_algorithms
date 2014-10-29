
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "cl_utils.h"
#include "time_ms.h"

// allows the user to specify the problem size and platform at compile time
#ifndef N
	#define N 1024
#endif
#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif
#ifndef IT
	#define IT 100
#endif

#define KERNEL_FILE_NAME "./jacobi.cl"


#define VALUE float

VALUE u[N][N], f[N][N];

VALUE init_func(int x, int y) {
	return 40 * sin((VALUE)(16 * (2 * x - 1) * y));
}

void print_result(VALUE res[N][N]) {
	printf("Result:\n");
	for(int i = 1; i < N-1; ++i) {
		printf("[");
		for(int j = 1; j < N-1; ++j) {
			printf(" %3.5f", res[i][j]);
		}
		printf(" ]\n");
	}
}

double getDurationMS(cl_event event) {
	cl_ulong start = 0;
	cl_ulong end = 0;
	cl_int ret;

	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), (void *)&start, NULL);
	ret |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void *)&end, NULL);

	CLU_ERRCHECK(ret, "Unable to read profiling information!");

	// the values are measured in nano seconds!
	return (double)(end - start) / 1000000.0;
}

int main()
{	
//	unsigned long start_time = time_ms();

	// init matrix
	memset(u, 0, N*N);

	printf("Jacobi with  N=%d\n", N);
	printf("Jacobi with IT=%d\n", IT);

	// init F
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			f[i][j] = init_func(i, j);

	VALUE factor = pow((VALUE)1/N, 2);

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
	
	// create memory buffers
	cl_int err;
	cl_mem matrix_U = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(VALUE), NULL, &err);
	cl_mem matrix_F = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(VALUE), NULL, &err);
	cl_mem matrix_TMP = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(VALUE), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create buffer for matrix");

	// used for profiling info
	cl_event ev_write_U;
	cl_event ev_write_F;
	cl_event ev_kernel;
	cl_event ev_read_TMP;

	double write_total, read_total, kernel_total;
	write_total = read_total = kernel_total = 0.0;

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DN=%i -DVALUE=%s", N, EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
	CLU_ERRCHECK(err, "Failed to create matrix_mul kernel from program");

	/* ---------------------------- main part ----------------------------------- */

	// also initialize target matrix with zero values!!!
	err = clEnqueueWriteBuffer(command_queue, matrix_TMP, CL_TRUE, 0, N * N * sizeof(VALUE), u, 0, NULL, &ev_write_U);
	CLU_ERRCHECK(err, "Failed to write matrix to device");
	// write f to device
	err = clEnqueueWriteBuffer(command_queue, matrix_F, CL_FALSE, 0, N * N * sizeof(VALUE), f, 0, NULL, &ev_write_F);
	CLU_ERRCHECK(err, "Failed to write matrix F to device");

	// write matrix u to device
	err = clEnqueueWriteBuffer(command_queue, matrix_U, CL_FALSE, 0, N * N * sizeof(VALUE), u, 0, NULL, &ev_write_U);
	CLU_ERRCHECK(err, "Failed to write matrix U to device");

	// define global work size
	size_t size[2] = {N-2, N-2}; // two dimensional range

	cl_mem buffer_u;
	cl_mem buffer_tmp;

	for (int i = 0; i < IT; ++i) {
		// swap U and TMP arguments based on iteration counter
		if(i % 2 == 0) {
			buffer_u = matrix_U;
			buffer_tmp = matrix_TMP;
		} else {
			buffer_u = matrix_TMP;
			buffer_tmp = matrix_U;
		}
		// set kernel arguments
		cluSetKernelArguments(kernel, 4,
							  sizeof(cl_mem), (void *)&buffer_u,
							  sizeof(cl_mem), (void *)&matrix_F,
							  sizeof(cl_mem), (void *)&buffer_tmp,
							  sizeof(VALUE), (void *)&factor);

		// execute kernel
		err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, size, NULL, 0, NULL, &ev_kernel);
		CLU_ERRCHECK(err, "Failed to enqueue 2D kernel");
		// wait until execution completes
		clWaitForEvents(1, &ev_kernel);
		// add profiling information
		kernel_total += getDurationMS(ev_kernel);
	}

	// copy results back to host
	err = clEnqueueReadBuffer(command_queue, buffer_tmp, CL_TRUE, 0, N * N * sizeof(VALUE), u, 0, NULL, &ev_read_TMP);
	CLU_ERRCHECK(err, "Failed reading back result");

	// compute profiling information
	write_total += getDurationMS(ev_write_U);
	write_total += getDurationMS(ev_write_F);
	read_total += getDurationMS(ev_read_TMP);


	/* ---------------------------- evaluate results ---------------------------------- */
	// print result
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
//	printf("Verification: %4s\n", (success) ? "OK" : "ERR");

	printf("Write total:      %9.4f ms\n", write_total);
	printf("Read total:       %9.4f ms\n", read_total);
	printf("Kernel execution: %9.4f ms\n", kernel_total);
	printf("Time total:       %9.4f ms\n\n", write_total + read_total + kernel_total);
//	print_result(u);


	/* ---------------------------- finalization ------------------------------------- */

	err = clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(matrix_U);
	err |= clReleaseMemObject(matrix_F);
	err |= clReleaseMemObject(matrix_TMP);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
	
	return EXIT_SUCCESS;
}

