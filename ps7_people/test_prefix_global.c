#include "prefixglobal.h"
#include "time_ms.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif
#ifndef LOCAL_SIZE
	#define LOCAL_SIZE 128
#endif

void print_array(int *arr, int n) {
	printf("[");
	for (int i = 0; i < n; ++i) {
		printf(" %d", arr[i]);
	}
	printf(" ]\n");
}

bool validate_result(int *result, int n) {
	for(int i = 0; i < n; ++i)
		if(result[i] != i) {
			printf("ERROR: Element '%d' is '%d'!\n", i, result[i]);
			return false;
		}

	return true;
}

int main(int argc, char *argv[])
{
	int elems = 1000000;
	if(argc > 1)
		elems = atof(argv[1]);

	printf("Computing prefix sum for n=%d\n", elems);

	// allocate memory for data set
	int *data = (int*)malloc(elems*sizeof(int));

	for(int i=0; i<elems; ++i) {
		data[i] = 1;
	}
//	printf("Input:  "); print_array(data, elems);

	int *result = (int*)malloc(elems*sizeof(int));

	init_prefix_cl(CL_DEVICE, LOCAL_SIZE);

	unsigned long long start_time = time_ms();
	prefix_sum(data, result, elems);
	unsigned long long total_time = time_ms() - start_time;

	// print result
	printf("Done, took %12llu ms\n", total_time);
	printf("Validation: %s\n", validate_result(result, elems) ? "OK" : "Not correct!");
//	printf("Output: "); print_array(result, elems);

	// finalization
	finalize_prefix_cl();

	free(data);
	free(result);

	return EXIT_SUCCESS;
}

