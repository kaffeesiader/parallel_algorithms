
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>

#include "cl_utils.h"

// allows the user to specify the problem size and platform at compile time
#ifndef N
	#define N 1000
#endif
#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif

#define M N
#define K N

#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define MAX(X,Y) ((X)>(Y)?(X):(Y))

#define VALUE float

#define KERNEL_FILE_NAME "./matrix_mul.cl"

// create the matrices
VALUE A[N][M];
VALUE B[M][K];
VALUE C[N][K];

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

int main()
{	
	// A contains real values
	for(int i=0; i<N; i++) {
		for(int j=0; j<M; j++) {
			A[i][j] = (VALUE)(i*j);
		}
	}

	// B is the identity matrix
	for(int i=0; i<M; i++) {
		for(int j=0; j<K; j++) {
			B[i][j] = (VALUE)((i==j)?1:0);
		}
	}

	// ocl initialization
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);
	
	// create memory buffers
	cl_int err;
	cl_mem matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY, N * M * sizeof(VALUE), NULL, &err);
	cl_mem matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY, M * K * sizeof(VALUE), NULL, &err);
	cl_mem matrix_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * K * sizeof(VALUE), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create buffer for matrix");

	// used for profiling info
	cl_event ev_write_A;
	cl_event ev_write_B;
	cl_event ev_kernel;
	cl_event ev_read_C;

	// fill memory buffers
	err = clEnqueueWriteBuffer(command_queue, matrix_A, CL_FALSE, 0, N * M * sizeof(VALUE), A, 0, NULL, &ev_write_A);
	err |= clEnqueueWriteBuffer(command_queue, matrix_B, CL_TRUE, 0, M * K * sizeof(VALUE), B, 0, NULL, &ev_write_B);
	CLU_ERRCHECK(err, "Failed to write matrix to device");

	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DN=%i -DK=%i -DM=%i -DVALUE=%s", N, K, M, EXPAND_AND_QUOTE(VALUE));
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, tmp);
	cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
	CLU_ERRCHECK(err, "Failed to create matrix_mul kernel from program");

	// set arguments and execute kernel
	size_t size[2] = {N, M}; // two dimensional range
	cluSetKernelArguments(kernel, 3, sizeof(cl_mem), (void *)&matrix_A, sizeof(cl_mem), (void *)&matrix_B, sizeof(cl_mem), (void *)&matrix_C);
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, size, NULL, 0, NULL, &ev_kernel), "Failed to enqueue 2D kernel");

	// copy results back to host
	err = clEnqueueReadBuffer(command_queue, matrix_C, CL_TRUE, 0, N * K * sizeof(VALUE), C, 0, NULL, &ev_read_C);
	CLU_ERRCHECK(err, "Failed reading back result");

	// check result
	int success = 1;	
	for (int i=0; i<N; i++) {
		for (int j=0; j<MIN(M,K); j++) {
			if (A[i][j] != C[i][j]) {
				success = 0;
			}
		}
		for (int j=MIN(M,K); j<MAX(M,K); j++) {
			if (C[i][j] != 0) {
				success = 0;
			}
		}
	}

	// print result
	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
	printf("Verification: %4s\n", (success) ? "OK" : "ERR");

	printf("Write A:          %9u ms\n", getDurationMS(ev_write_A));
	printf("Write B:          %9u ms\n", getDurationMS(ev_write_B));
	printf("Kernel execution: %9u ms\n", getDurationMS(ev_kernel));
	printf("Read C:           %9u ms\n", getDurationMS(ev_read_C));

	cl_uint duration = getDurationMS(ev_write_A)
			+ getDurationMS(ev_write_B)
			+ getDurationMS(ev_kernel)
			+ getDurationMS(ev_read_C);

	printf("Time total:       %9u ms\n\n", duration);

	// finalization
	err = clFinish(command_queue);
	err |= clReleaseKernel(kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(matrix_A);
	err |= clReleaseMemObject(matrix_B);
	err |= clReleaseMemObject(matrix_C);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
	
	return 0;
}

