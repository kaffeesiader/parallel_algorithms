export AMDOCLSDK=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64
g++ hello.cpp -I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL -o hello
g++ info.cpp -I$AMDOCLSDK/include -L$AMDOCLSDK/lib/x86_64 -lOpenCL -o info
