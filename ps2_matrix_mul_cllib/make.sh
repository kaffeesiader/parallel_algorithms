export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${N+set}" != set ; then
    export N=1024
fi

if test "${CL_PLAT+set}" != set ; then
    export CL_PLAT=0
fi

# gcc -O3 -std=c99 -Wall -Werror matrix_mul.c -o matrix_mul_N$N -DN=$N 
# gcc -O3 -std=c99 -Wall -Werror -fopenmp matrix_mul_omp.c -o matrix_mul_omp_N$N -DN=$N
gcc -O3 -std=c99 -Wall -Werror matrix_mul_ocl.c $OCLLIB -o matrix_mul_ocl_N${N}_P$CL_PLAT -DN=$N -DCL_DEVICE=$CL_PLAT -DCL_USE_DEPRECATED_OPENCL_2_0_APIS