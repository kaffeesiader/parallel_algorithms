
#include "prefixglobal.h"
#include "cl_utils.h"

#define KERNEL_FILE_NAME "./prefixglobal.cl"

cl_command_queue _command_queue;
cl_program _program;
cl_kernel _scan_kernel, _add_kernel;
cl_context _context;
cl_device_id _device_id;
int _local_size;

bool _initialized;

void show_array(int *arr, int n) {
	printf("[");
	for (int i = 0; i < n; ++i) {
		printf(" %d", arr[i]);
	}
	printf(" ]\n");
}

bool val_prefix_sum(int *input, int *output, int n) {
	int tmp = 0;
	for(int i = 0; i < n; ++i) {
	      if(!output[i] == tmp)
		return false;
	      tmp += input[i];
	}
	return true;
}

void prefix_sum(int *input, int *output, int n) {

	if(!_initialized) {
		fprintf(stderr, "OpenCL not initialized!");
		exit(EXIT_FAILURE);
	}

	// ensure that global size is at least local size
	// and is devideable by local size...
	int global_size = n;
	if((global_size % _local_size) > 0) {
		global_size += (_local_size - (global_size % _local_size));
	}

	int sum_buffer_size = global_size / _local_size;

	size_t g_sz = global_size;
	size_t l_sz = _local_size;

	printf("Local size: %d, Global size: %d, Sum buffer size: %d\n", _local_size, global_size, sum_buffer_size);

	// create memory buffers
	cl_int err;
	cl_mem mem_input = clCreateBuffer(_context, CL_MEM_READ_ONLY, global_size * sizeof(int), NULL, &err);
	cl_mem mem_output = clCreateBuffer(_context, CL_MEM_READ_WRITE, global_size * sizeof(int), NULL, &err);

	cl_mem mem_sum_buffer = clCreateBuffer(_context, CL_MEM_READ_WRITE, sum_buffer_size * sizeof(int), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create memory buffers");

	// fill memory buffers
	err = clEnqueueWriteBuffer(_command_queue, mem_input, CL_FALSE, 0, n * sizeof(int), input, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	// set arguments and execute 'scan' kernel
	cluSetKernelArguments(_scan_kernel, 4,
				  sizeof(cl_mem), (void *)&mem_input,
				  sizeof(cl_mem), (void *)&mem_output,
				  sizeof(int), (void *)&n,
				  sizeof(cl_mem), (void *)&mem_sum_buffer);

	err = clEnqueueNDRangeKernel(_command_queue, _scan_kernel, 1, NULL, &g_sz, &l_sz, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to enqueue 'scan' kernel");

	if(n > _local_size) {
		int *sum_buffer = malloc(sum_buffer_size * sizeof(int));
		err = clEnqueueReadBuffer(_command_queue, mem_sum_buffer, CL_TRUE, 0, sum_buffer_size*sizeof(int), sum_buffer, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to read sum buffer");

		// compute the prefix sum of our sum buffer...
		prefix_sum(sum_buffer, sum_buffer, sum_buffer_size);
		
		err = clEnqueueWriteBuffer(_command_queue, mem_sum_buffer, CL_TRUE, 0, sum_buffer_size * sizeof(int), sum_buffer, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to write data to device");

		// ... and apply the results to our block sums
		// set arguments and execute 'add' kernel
		cluSetKernelArguments(_add_kernel, 3,
					  sizeof(cl_mem), (void *)&mem_output,
					  sizeof(cl_mem), (void *)&mem_sum_buffer,
					  sizeof(int), (void *)&n);

		err = clEnqueueNDRangeKernel(_command_queue, _add_kernel, 1, NULL, &g_sz, &l_sz, 0, NULL, NULL);
		CLU_ERRCHECK(err, "Failed to enqueue 'add' kernel");
		free(sum_buffer);
	}

	// read result
	err = clEnqueueReadBuffer(_command_queue, mem_output, CL_TRUE, 0, n*sizeof(int), output, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to read result");
	
	err |= clReleaseMemObject(mem_input);
	err |= clReleaseMemObject(mem_output);
	err |= clReleaseMemObject(mem_sum_buffer);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
}

void init_prefix_cl(int device, int local_size)
{
	if(_initialized)
		return;

	// ocl initialization
	_device_id = cluInitDevice(device, &_context, &_command_queue);
	_local_size = local_size;

	printf("OCL Device: %s\n", cluGetDeviceDescription(_device_id, CL_DEVICE));

	cl_int err;
	// create kernel from source
	char tmp[1024];
	sprintf(tmp, "-DGROUPSIZE=%i", local_size);
	_program = cluBuildProgramFromFile(_context, _device_id, KERNEL_FILE_NAME, tmp);
	_scan_kernel = clCreateKernel(_program, "scan", &err);
	_add_kernel = clCreateKernel(_program, "add", &err);
	CLU_ERRCHECK(err, "Failed to create kernels from program");

	_initialized = true;
}


void finalize_prefix_cl()
{
	if(!_initialized)
		return;

	cl_int err;
	// finalization
	err = clFinish(_command_queue);
	err |= clReleaseKernel(_scan_kernel);
	err |= clReleaseKernel(_add_kernel);
	err |= clReleaseProgram(_program);
	err |= clReleaseCommandQueue(_command_queue);
	err |= clReleaseContext(_context);

	CLU_ERRCHECK(err, "Failed during ocl cleanup");

	_initialized = false;
}
