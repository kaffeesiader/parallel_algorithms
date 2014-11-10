export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
# export AMDOCLSDK=/software/AMD/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi

#gcc -O3 -Wall -Werror -std=c99 $OCLLIB search_cl.c dSFMT.c -o search_cl_D$CL_DEVICE -DCL_DEVICE=$CL_DEVICE -lm
gcc -O3 -Wall -Werror -std=c99 search.c dSFMT.c -o search -lm

#gcc -O3 -Wall -Werror -std=c99 search_omp.c dSFMT.c -o search_omp -lm -fopenmp
#gcc -O3 -Wall -Werror -std=c99 search_omp_seq.c -g dSFMT.c -o search_omp_seq -lm -fopenmp
