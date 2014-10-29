
echo "Starting test run..."

echo -e "\nDevice 0, IT=100, N=2048"
./jacobi_ocl_V1_N2048_IT100_P0
echo "Device 0, IT=100, N=2048"
./jacobi_ocl_V2_N2048_IT100_P0
echo "Device 1, IT=100, N=2048"
./jacobi_ocl_V1_N2048_IT100_P1
echo -e "\nDevice 1, IT=100, N=2048"
./jacobi_ocl_V2_N2048_IT100_P1

echo -e "\nTest run completed."



