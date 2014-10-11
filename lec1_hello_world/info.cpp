#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define cl_check_status(_expr) 								\
	do {										\
		cl_int status = _expr;							\
	if (status == CL_SUCCESS)							\
			break;								\
		else									\
			fprintf(stderr,"OpenCL Error while calling %s'\n",#_expr);	\
			abort();							\
	} while (0);									\


int main() {

	//it is a good practice to check the status of each API call
	cl_int status;
	
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;
	

	//---------------------------------------------------------//
	//--------------retrieving platforms-----------------------//	
	// the number of platforms is retrieved by using a first call
	// to clGetPlatformsIDs() with NULL argument as second argument
	status 		= clGetPlatformIDs(0,NULL,&numPlatforms);
	platforms	= (cl_platform_id *)malloc(
			numPlatforms * sizeof(cl_platform_id));

	printf("Number of platforms: %d\n",numPlatforms);
	// the second call, get the platforms
	status 		= clGetPlatformIDs(numPlatforms,platforms, NULL);


	//--------------------------------------------------------//
	//---------printing_platforms_information-----------------//
	//--------------------------------------------------------//
	
	for (int i = 0; i < numPlatforms; i++) {
		//-------------retrieving devices ------------------------//
		cl_uint numDevices 	= 0;						
		cl_check_status(clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_ALL, 0, NULL,&numDevices));
		printf("Total number of devices: %d\n",numDevices);
		

		char buffer[10240];
		printf("  More information about platform -- %d --\n", i);
		cl_check_status(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		cl_check_status(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		cl_check_status(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		cl_check_status(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		cl_check_status(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}

	//--------------------------------------------------------//
	
	//free host resources	
	free(platforms);
	return 0;
}
