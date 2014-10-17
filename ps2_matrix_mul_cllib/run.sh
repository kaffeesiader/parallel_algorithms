echo "Starting test run..."

echo -e "\nDevice 0, N=1024"
./matrix_mul_ocl_N1024_P0
echo "Device 0, N=2048"
./matrix_mul_ocl_N2048_P0
echo "Device 0, N=4096"
./matrix_mul_ocl_N4096_P0

echo -e "\nDevice 1, N=1024"
./matrix_mul_ocl_N1024_P1
echo "Device 1, N=2048"
./matrix_mul_ocl_N2048_P1
echo "Device 1, N=4096"
./matrix_mul_ocl_N4096_P1

echo -e "\nTest run completed."