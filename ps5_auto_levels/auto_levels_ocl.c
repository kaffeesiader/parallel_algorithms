
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "cl_utils.h"
#include "time_ms.h"

// allows the user to specify the platform at compile time
#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif

#define KERNEL_FILE_NAME "./auto_levels.cl"
//#define DEBUG
#ifndef LOC_SZ
	#define LOC_SZ 256
#endif

#define MIN 0
#define MAX 1
#define AVG 2

void print_result(unsigned char *data, int N) {
	printf("Result: ");
	printf("[");
		for(int i = 0; i < N; ++i) {
			printf(" %d", data[i]);
		}
	printf(" ]\n");
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

int main(int argc, char **argv)
{
//	unsigned long start_time = time_ms();

	if(argc != 3) {
		printf("Usage: auto_levels_ocl [inputfile] [outputfile]\nExample: auto_levels test.png test_adjusted.png\n");
		return -1;
	}

	printf("Opening image '%s'...\n", argv[1]);

	// load image and determine image parameters
	int width, height, components;
	unsigned char *data = stbi_load(argv[1], &width, &height, &components, 0);

	printf("Image details: width=%d, height=%d, channels=%d\n", width, height, components);
	printf("Local size: %d\n", LOC_SZ);

	assert (((width*height) % LOC_SZ) == 0);

	int length = width * height;

	// for testing only
//	width = 2;
//	height = 2;
//	components = 1;



//	unsigned char data[] = {45, 45, 173, 173};

//	unsigned char data[length*components];

//	for (int i = 0; i < length; ++i) {
//		for (int c = 0; c < components; ++c) {
//			data[i*components+0] = (unsigned char)45;
//			data[i*components+1] = (unsigned char)177;
//		}
//	}

	// ocl initialization
	cl_int err;
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id = cluInitDevice(CL_DEVICE, &context, &command_queue);

	int image_buf_size = length * components;
	// compute the number of workgroups...
	int num_WG = length / LOC_SZ;
	int result_buf_size = 3 * components * num_WG; // 3(min, max, avg) * (number of workgroups) * components
	// create memory buffers
	cl_mem image_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, image_buf_size*sizeof(unsigned char), NULL, &err);
	// the result buffer has the size 3 (min,max,av) * number of workgroups(= global size / local size)
	cl_mem result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, result_buf_size*sizeof(unsigned char), NULL, &err);

	CLU_ERRCHECK(err, "Failed to create memory buffer for image");

	// used for profiling info
	cl_event ev_reduction;
	cl_event ev_adjustment;
	cl_event ev_read;
	cl_event ev_write;

	// create kernels from source
	cl_program program = cluBuildProgramFromFile(context, device_id, KERNEL_FILE_NAME, NULL);
	cl_kernel reduce_kernel = clCreateKernel(program, "reduce_min_max_avg", &err);
	CLU_ERRCHECK(err, "Failed to create 'reduce' kernel from program");
	cl_kernel adjust_kernel = clCreateKernel(program, "adjust_image", &err);
	CLU_ERRCHECK(err, "Failed to create 'adjust_image'' kernel from program");

	// write image data to device
	err = clEnqueueWriteBuffer(command_queue, image_buf, CL_FALSE, 0, image_buf_size*sizeof(unsigned char), data, 0, NULL, &ev_write);
	CLU_ERRCHECK(err, "Failed to write image to device");

	/* ---------------------------- reduction part ----------------------------------- */

	// define work size
	size_t g_work_size[] = {length};
	size_t l_work_size[] = {LOC_SZ};

	// set kernel arguments
	cluSetKernelArguments(reduce_kernel, 5,
						  sizeof(cl_mem), (void *)&image_buf,
						  // local memory buffer
						  3*components*LOC_SZ*sizeof(unsigned char), NULL,
						  sizeof(int), (void *)&length,
						  sizeof(int), (void *)&components,
						  sizeof(cl_mem), (void *)&result_buf);

	// execute reduce kernel
	err = clEnqueueNDRangeKernel(command_queue, reduce_kernel, 1, NULL, g_work_size, l_work_size, 0, NULL, &ev_reduction);
	CLU_ERRCHECK(err, "Failed to enqueue reduction kernel");

	// copy results back to host
	unsigned char mma_result[result_buf_size];
	err = clEnqueueReadBuffer(command_queue, result_buf, CL_TRUE, 0, result_buf_size*sizeof(unsigned char), mma_result, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed reading back result");

#ifdef DEBUG
	print_result(mma_result, result_buf_size);
#endif

	/* ----------------------------- combine the results of the different work groups -------------------------*/

	unsigned char min_val[components], max_val[components], avg_val[components];
	float min_fac[components], max_fac[components];
	// combine the results of the different working groups per color component
	for(int c = 0; c < components; ++c) {
		unsigned char min = 255;
		unsigned char max = 0;
		int sum = 0;
		// iterate over work group results
		for (int w = 0; w < num_WG; ++w) {
			// compute offset within results array
			int offset = 3*w*components + 3*c;

			min = (mma_result[offset+MIN] > min) ? min : mma_result[offset+MIN];
			max = (mma_result[offset+MAX] < max) ? max : mma_result[offset+MAX];
			sum += mma_result[offset+AVG];
		}

		min_val[c] = min;
		max_val[c] = max;

		// calculate average and multiplicative factors
		avg_val[c] = (unsigned char)(sum / num_WG);
		min_fac[c] = (float)avg_val[c] / ((float)avg_val[c] - (float)min_val[c]);
		max_fac[c] = (255.0f-(float)avg_val[c]) / ((float)max_val[c] - (float)avg_val[c]);
		printf("Component %1u: %3u/%3u/%3u * %3.2f/%3.2f\n", c, min_val[c],avg_val[c],max_val[c], min_fac[c],max_fac[c]);
	}

	/* ----------------------------- adjust the image --------------------------------- */
	// create additional memory buffers
	cl_mem mem_avg_val = clCreateBuffer(context, CL_MEM_READ_ONLY, components*sizeof(unsigned char), NULL, &err);
	cl_mem mem_min_fac = clCreateBuffer(context, CL_MEM_READ_ONLY, components*sizeof(float), NULL, &err);
	cl_mem mem_max_fac = clCreateBuffer(context, CL_MEM_READ_ONLY, components*sizeof(float), NULL, &err);

	err = clEnqueueWriteBuffer(command_queue, mem_avg_val, CL_FALSE, 0, components*sizeof(unsigned char), avg_val, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(command_queue, mem_min_fac, CL_FALSE, 0, components*sizeof(float), min_fac, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(command_queue, mem_max_fac, CL_FALSE, 0, components*sizeof(float), max_fac, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed writing to memory buffer ");
	// set kernel arguments
	cluSetKernelArguments(adjust_kernel, 5,
						  sizeof(cl_mem), (void *)&image_buf,
						  sizeof(cl_mem), (void *)&mem_avg_val,
						  sizeof(cl_mem), (void *)&mem_min_fac,
						  sizeof(cl_mem), (void *)&mem_max_fac,
						  sizeof(int), (void *)&components);

	// execute image adjustment kernel (no need to define a local work size...)
	err = clEnqueueNDRangeKernel(command_queue, adjust_kernel, 1, NULL, g_work_size, NULL, 0, NULL, &ev_adjustment);
	CLU_ERRCHECK(err, "Failed to enqueue image adjustment kernel");

	// copy modified image back to host
	err = clEnqueueReadBuffer(command_queue, image_buf, CL_TRUE, 0, image_buf_size*sizeof(unsigned char), data, 0, NULL, &ev_read);
	CLU_ERRCHECK(err, "Failed reading back result");

//	print_result(data, 4);


	/* ---------------------------- evaluate results ---------------------------------- */

	// add profiling information
	double reduction_time = getDurationMS(ev_reduction);
	double adjustment_time = getDurationMS(ev_adjustment);
	double read_time = getDurationMS(ev_read);
	double write_time = getDurationMS(ev_write);

	printf("OCL Device: %s\n", cluGetDeviceDescription(device_id, CL_DEVICE));
	printf("Reduction kernel : %9.4f ms\n", reduction_time);
	printf("Adjustment kernel: %9.4f ms\n", adjustment_time);
	printf("Write image      : %9.4f ms\n", write_time);
	printf("Read image       : %9.4f ms\n", read_time);
	printf("Time total       : %9.4f ms\n\n", reduction_time + adjustment_time, read_time, write_time);

	printf("\nAdjustment completed - writing image '%s'...\n", argv[2]);
	stbi_write_png(argv[2], width, height, components, data, width*components);
	stbi_image_free(data);
	printf("Done!\n");


	/* ---------------------------- finalization ------------------------------------- */

	err = clFinish(command_queue);
	err |= clReleaseKernel(reduce_kernel);
	err |= clReleaseKernel(adjust_kernel);
	err |= clReleaseProgram(program);
	err |= clReleaseMemObject(image_buf);
	err |= clReleaseMemObject(result_buf);
	err |= clReleaseMemObject(mem_avg_val);
	err |= clReleaseMemObject(mem_min_fac);
	err |= clReleaseMemObject(mem_max_fac);
	err |= clReleaseCommandQueue(command_queue);
	err |= clReleaseContext(context);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");

	return EXIT_SUCCESS;
}

