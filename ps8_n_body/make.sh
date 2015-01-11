export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
export OCLLIB="-I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL"

if test "${CL_DEVICE+set}" != set ; then
    export CL_DEVICE=0
fi
if test "${N+set}" != set ; then
    export N=200
fi

gcc -O3 -Wall -Werror -std=c99 n_body.c -o n_body -DN=$N -lglut -lm -lGL -lGLU -lGLEW
gcc -O3 -Wall -Werror -std=c99 n_body_cl.c $OCLLIB -o n_body_P${CL_DEVICE} -DN=$N -DCL_DEVICE=$CL_DEVICE -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -lglut -lm -lGL -lGLU -lGLEW $1
