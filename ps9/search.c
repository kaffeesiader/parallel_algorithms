#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MAX_VAL 101

void printArray(int *array, int n) {
	printf("[ ");
	for (int i = 0; i < n; ++i) {
		printf("%d ", array[i]);
	}
	printf("]\n");
}

int *createArray(int n) {
	int *array = malloc(n * sizeof(int));

	if(array == NULL) {
		printf("Unable to allocate array memory!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < n; ++i)
		array[i] = rand() % MAX_VAL;

	return array;
}

int indexOf(int value, int *array, int lo, int hi) {
	int mid = lo + (hi-lo)/2;

	if(array[mid] == value) {
		return mid;
	} else if(hi == lo) {
		return -1;
	} else {
		int res_lo, res_hi;

		#pragma omp task shared(res_lo)
		res_lo = indexOf(value, array, lo, mid);

		#pragma omp task shared(res_hi)
		res_hi = indexOf(value, array, mid+1, hi);

		#pragma omp taskwait
		if(res_lo >= 0)
			return res_lo;
		else
			return res_hi;
	}
}

int main(int argc, char **argv) {

	if(argc < 3) {
		printf("USAGE: search [NUM] [N]\n");
		return EXIT_FAILURE;
	}

	int n = atoi(argv[1]), value = atoi(argv[2]), index;

	srand(time(NULL));

	int *array = createArray(n);

	#pragma omp parallel
	{
		#pragma omp single
		index = indexOf(value, array, 0, n-1);
	}

	if(index < 0)
		printf("Value '%d' not found within array\n", value);
	else {
		assert(array[index] == value);
		printf("Found value '%d' at index %d\n", value, index);
	}

//	printArray(array, n);

	free(array);

	return EXIT_SUCCESS;
}

