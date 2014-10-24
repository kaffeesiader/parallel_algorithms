#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "time_ms.h"

// allows the user to specify the problem size at compile time
#ifndef N
	#define N 1024
#endif
#ifndef IT
	#define IT 100
#endif

#define VALUE double

VALUE u[N][N], tmp[N][N], f[N][N];

VALUE init_func(int x, int y) {
	return 40 * sin((VALUE)(16 * (2 * x - 1) * y));
}

void print_result(VALUE res[N][N]) {
	for(int i = 1; i < N-1; ++i) {
		printf("[");
		for(int j = 1; j < N-1; ++j) {
			printf(" %3.5f", res[i][j]);
		}
		printf(" ]\n");
	}
}

int main(int argc, char** argv) {
	unsigned long start_time = time_ms();

	printf("Jacobi with  N=%d\n", N);
	printf("Jacobi with IT=%d\n", IT);

	// init matrix
	memset(u, 0, N*N);

	// init F
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			f[i][j] = init_func(i, j);

	VALUE factor = pow((VALUE)1/N, 2);

	for(int it=0; it<IT; it++) {
		// main Jacobi loop
		#ifdef OMP
		#pragma omp parallel for
		#endif
		for(int i=1; i < N-1; i++) {
			for(int j=1; j < N-1; j++) {
				tmp[i][j] = (VALUE)1/4 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] - factor * f[i][j]);
			}
		}

		memcpy(u, tmp, N*N*sizeof(double));

	}
	
	printf("Time: %9lu ms\n", time_ms() - start_time);
//	print_result(tmp);
}

