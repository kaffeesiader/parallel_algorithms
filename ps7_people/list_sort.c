#include "people.h"

void exclusive_prefix_sum(int *input, int n) {
	int sum = 0, tmp = 0;
	for(int i = 0; i < n; ++i) {
		tmp = input[i];
		input[i] = sum;
		sum += tmp;
	}
}

void count_sort(person_t **persons, person_t **sorted, int n) {

	int C[MAX_AGE];
	memset(C, 0, MAX_AGE*sizeof(int));
	// compute histogram
	for (int i = 0; i < n; ++i)
		C[persons[i]->age]++;

	exclusive_prefix_sum(C, MAX_AGE);

	for (int i = 0; i < n; ++i) {
		int value = persons[i]->age;
		int idx = C[value]++;
		sorted[idx] = persons[i];
	}
}

int main(int argc, char *argv[])
{
	if(argc != 3) {
		printf("Usage: list_gen [N] [SEED]\nExample: search 10000 1\n");
		return EXIT_FAILURE;
	}

	unsigned int num = atoi(argv[1]); unsigned int seed = atoi(argv[2]);
	srand(seed);

	// allocate memory for data set
	person_t *persons[num];
	person_t *sorted[num];

	generate_list(persons, num);
	printf("--- UNSORTED LIST --- \n");
	print_list(persons, num);

	count_sort(persons, sorted, num);

	printf("--- SORTED LIST --- \n");
	print_list(sorted, num);

	free_list(persons, num);

	return EXIT_SUCCESS;
}
