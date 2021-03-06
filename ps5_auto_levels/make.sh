export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi
if test "${L+set}" != set ; then
    export L=256
fi

#gcc -O2 -std=c99 $OCLLIB auto_levels_cl.c stb_image.c stb_image_write.c -o auto_levels_cl_D$CL_DEVICE -DCL_DEVICE=$CL_DEVICE -lm
gcc -O2 -std=c99 auto_levels.c stb_image.c stb_image_write.c -o auto_levels -lm
gcc -O3 -std=c99 auto_levels_ocl.c stb_image.c stb_image_write.c $OCLLIB -o auto_levels_ocl_L${L}_P${CL_DEVICE} -DLOC_SZ=$L -DCL_DEVICE=$CL_DEVICE -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lm $1

#gcc -std=c99 -Wall -Werror test.c -o test
