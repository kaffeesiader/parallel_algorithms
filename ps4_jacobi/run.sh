
echo "Starting test run..."

echo -e "\nV2 - Device GPU, IT=100, N=2048"
./jacobi_ocl_V2_N2048_IT100_P0
echo "V2 - Device CPU, IT=100, N=2048"
./jacobi_ocl_V2_N2048_IT100_P1
echo "V3 - Device GPU, IT=100, N=2048, L=4"
./jacobi_ocl_V3_N2048_L4_IT100_P0
echo -e "V3 - Device CPU, IT=100, N=2048, L=4"
./jacobi_ocl_V3_N2048_L4_IT100_P1

echo -e "\nTest run completed."



