export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${N+set}" != set ; then
    export N=2048
fi
if test "${L+set}" != set ; then
    export L=8
fi
if test "${IT+set}" != set ; then
    export IT=100
fi
if test "${CL_PLAT+set}" != set ; then
    export CL_PLAT=0
fi

# gcc -O3 -std=c99 -Wall -Werror jacobi.c -o jacobi_N${N}_IT${IT} -DN=$N -DIT=$IT -lm
# gcc -O3 -std=c99 -Wall -Werror -fopenmp jacobi.c -o jacobi_omp_N${N}_IT${IT} -DOMP=1 -DN=$N -DIT=$IT -lm
gcc -O3 -std=c99 -Wall -Werror jacobi_ocl_v1.c $OCLLIB -o jacobi_ocl_V1_N${N}_IT${IT}_P${CL_PLAT} -DN=$N -DIT=$IT -DCL_DEVICE=$CL_PLAT -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lm
gcc -O3 -std=c99 -Wall -Werror jacobi_ocl_v2.c $OCLLIB -o jacobi_ocl_V2_N${N}_IT${IT}_P${CL_PLAT} -DN=$N -DIT=$IT -DCL_DEVICE=$CL_PLAT -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lm
gcc -O3 -std=c99 -Wall -Werror jacobi_ocl_v3.c $OCLLIB -o jacobi_ocl_V3_N${N}_L${L}_IT${IT}_P${CL_PLAT} -DN=$N -DL_SZ=$L -DIT=$IT -DCL_DEVICE=$CL_PLAT -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lm $1
