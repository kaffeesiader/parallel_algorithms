AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
OCLLIB=-I${AMDOCLSDK}/include -L${AMDOCLSDK}/lib/x86_64 -lOpenCL

OBJDIR=bin
SRCDIR=src

ifndef CL_PLAT
	CL_PLAT=0
endif

ifndef N
	N=1024
endif

all:
	@mkdir -p $(OBJDIR)
	@cp ${SRCDIR}/matrix_mul_ocl.cl ${OBJDIR}
	
	gcc -O3 -std=c99 -Wall -Werror ${SRCDIR}/matrix_mul.c -o $(OBJDIR)/matrix_mul_N${N} -DN=$(N) 
	gcc -O3 -std=c99 -Wall -Werror -fopenmp ${SRCDIR}/matrix_mul_omp.c -o $(OBJDIR)/matrix_mul_omp_N${N} -DN=${N}
	gcc -O3 -std=c99 -Wall -Werror ${SRCDIR}/matrix_mul_ocl.c ${OCLLIB} -o $(OBJDIR)/matrix_mul_ocl_N${N}_P${CL_PLAT} -DN=${N} -DCL_USE_DEPRECATED_OPENCL_2_0_APIS

clean:
	rm -r bin
	
