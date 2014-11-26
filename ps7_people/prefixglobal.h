#ifndef PREFIXGLOBAL_H
#define PREFIXGLOBAL_H

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>

void init_prefix_cl(int device, int local_size);
void prefix_sum(int *input, int *output, int n);
void finalize_prefix_cl();

#endif // PREFIXGLOBAL_H
