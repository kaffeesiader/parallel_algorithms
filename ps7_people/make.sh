export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
# export AMDOCLSDK=/software/AMD/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi


gcc -O3 -Wall -Werror -std=c99 list_gen.c -o list_gen
gcc -g -Wall -Werror -std=c99 list_sort.c -o list_sort
#gcc -O3 -Wall -Werror -std=c99 search_ocl.c dSFMT.c $OCLLIB -o search_cl_P${CL_DEVICE} -DCL_DEVICE=$CL_DEVICE -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lm $1
