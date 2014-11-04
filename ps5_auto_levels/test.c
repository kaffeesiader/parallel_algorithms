#include <stdio.h>

int main(int argc, char *argv[])
{
	for(int i = 1; i <= 8; i <<= 1) {
		printf("%d\n", i);
	}
	return 0;
}

