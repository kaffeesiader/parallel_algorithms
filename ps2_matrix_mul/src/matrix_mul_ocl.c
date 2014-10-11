#include <stdio.h>
#include <stdlib.h>

#include "time_ms.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

// allows the user to specify the problem size at compile time
#ifndef N
	#define N 1000
#endif

#define M N
#define K N

#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define MAX(X,Y) ((X)>(Y)?(X):(Y))

#define VALUE double

// create the matices
VALUE A[N][M];
VALUE B[M][K];
VALUE C[N][K];

int main()
{
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int result;

	printf("Problem size N: %d\n", N);

	FILE *fp;
	char fileName[] = "./matrix_mul_ocl.cl";
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	/* Get Platform and Device Info */
	result = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

	result = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	/* Create OpenCL context */
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &result);

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &result);

	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,	(const size_t *)&source_size, &result);

	/* Build Kernel Program */
	result = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(result != CL_SUCCESS) {

		cl_build_status status;
		size_t logSize;
		char *programLog;

		// check build error and build status first
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
		// get log message size
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		// allocate memory for log message
		programLog = (char*) calloc (logSize+1, sizeof(char));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);

		printf("Build failed; error=%d, status=%d, programLog:\n%s", result, status, programLog);
		// release log message buffer
		free(programLog);
	}

	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, "vec_add", &result);


	unsigned long start_time = time_ms();

	/* ------------------ Initialize our matrizes ------------------*/
	// A contains real values
	for (int i=0; i<N; i++) {
		for (int j=0; j<M; j++) {
			A[i][j] = i*j;
		}
	}

	// B is the identity matrix
	for (int i=0; i<M; i++) {
		for (int j=0; j<K; j++) {
			B[i][j] = (i==j)?1:0;
		}
	}

	/* Create Memory Buffers for A, B and C */
	cl_mem mem_A;
	cl_mem mem_B;
	cl_mem mem_C;

	mem_A = clCreateBuffer(context, CL_MEM_READ_ONLY, N*M*sizeof(VALUE), NULL, &result);
	mem_B = clCreateBuffer(context, CL_MEM_READ_ONLY, M*K*sizeof(VALUE), NULL, &result);
	mem_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*K*sizeof(VALUE), NULL, &result);

	/* Write values into buffer */
	clEnqueueWriteBuffer(command_queue, mem_A, CL_TRUE, 0, N*M*sizeof(VALUE), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, mem_B, CL_TRUE, 0, M*K*sizeof(VALUE), B, 0, NULL, NULL);


	/* Set OpenCL Kernel Arguments */

	int m = M;
	int n = N;

	result = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_A);
	result = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_B);
	result = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_C);
	result = clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);
	result = clSetKernelArg(kernel, 4, sizeof(int), (void *)&m);

	/* Execute OpenCL Kernel */

	size_t global_work_size[] = {N, K};
	result = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

	if(result != CL_SUCCESS) {
		printf("Executing kernel not successful. RESULT: %d\n", result);
		return EXIT_FAILURE;
	}

	/* Copy results from the memory buffer */
	result = clEnqueueReadBuffer(command_queue, mem_C, CL_TRUE, 0, N*K*sizeof(VALUE), C, 0, NULL, NULL);

	// verify result
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

	// print verification result
	printf("Verification: %4s \n", (success) ? "OK" : "ERR");
	printf("Time: %9lu ms\n", time_ms() - start_time);


	/* Finalization */
	result = clFlush(command_queue);
	result = clFinish(command_queue);
	result = clReleaseKernel(kernel);
	result = clReleaseProgram(program);
	result = clReleaseMemObject(mem_A);
	result = clReleaseMemObject(mem_B);
	result = clReleaseMemObject(mem_C);
	result = clReleaseCommandQueue(command_queue);
	result = clReleaseContext(context);

	free(source_str);

	return EXIT_SUCCESS;
}

