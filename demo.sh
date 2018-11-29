make

function echo_header {
  printf "\n=================== %s ===================\n" "$1"
}

echo_header "MATRIX MULTIPLY"
echo "input:"
more tests/simple_3x3.txt
echo ""
bin/matrix_ops matmul run tests/simple_3x3.txt tests/simple_3x3.txt

echo_header "LU DECOMPOSITION"
echo "input:"
more tests/simple_10x10.txt
echo ""
bin/matrix_ops lu_decomposition run tests/simple_10x10.txt

echo_header "BLOCKED LU DECOMPOSITION"
echo "input:"
more tests/simple_10x10.txt
echo ""
bin/matrix_ops lu_blockeddecomposition run tests/simple_10x10.txt

echo_header "RANDOMIZED LU DECOMPOSITION"
echo "input:"
more tests/simple_10x10.txt
echo ""
bin/matrix_ops lu_randomizeddecomposition run tests/simple_10x10.txt

