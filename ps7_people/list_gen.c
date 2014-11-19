#include "people.h"

int main(int argc, char *argv[])
{
	if(argc != 3) {
		printf("Usage: list_gen [N] [SEED]\nExample: search 10000 1\n");
		return EXIT_FAILURE;
	}

	unsigned int elems = atoi(argv[1]); unsigned int seed = atoi(argv[2]);
	srand(seed);

	person_t *data[elems];
	generate_list(data,elems);
	print_list(data, elems);
	free_list(data, elems);

	return EXIT_SUCCESS;
}
