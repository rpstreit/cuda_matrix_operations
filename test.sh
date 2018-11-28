#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0 ]}" )" && pwd )"
TESTS=${DIR}/tests
ALL_PASS=true

MM_A_TESTS=(simple_3x2.txt simple_3x3.txt random1_1x32.txt ones_5x5.txt random1_7x7.txt random1_50x50.txt random1_64x64.txt random1_98x98.txt random1_200x200.txt)
MM_B_TESTS=(simple_2x3.txt simple_3x3.txt random1_32x32.txt random1_5x5.txt random1_7x4.txt random1_50x50.txt random1_64x64.txt random1_98x98.txt random1_200x200.txt)

LU_DECOMP_TESTS=(simple_3x3.txt simple_3x2.txt simple_4x4.txt simple_5x5.txt simple_8x8.txt simple_9x9.txt random1_7x4.txt random1_10x10.txt random1_17x17.txt random1_18x18.txt random1_19x19.txt random1_20x20.txt random1_27x27.txt random1_32x32.txt random1_37x37.txt random1_38x38.txt random1_39x39.txt random1_48x48.txt random1_50x50.txt random1_96x47.txt random1_98x98.txt random1_99x99.txt random1_100x100.txt random1_110x110.txt random1_121x121.txt random1_122x122.txt random1_128x128.txt random1_200x200.txt)
LU_SQDECOMP_TESTS=(simple_3x3.txt simple_4x4.txt simple_5x5.txt simple_8x8.txt simple_9x9.txt random1_10x10.txt random1_17x17.txt random1_18x18.txt random1_19x19.txt random1_20x20.txt random1_27x27.txt random1_32x32.txt random1_37x37.txt random1_38x38.txt random1_39x39.txt random1_48x48.txt random1_50x50.txt random1_98x98.txt random1_99x99.txt random1_100x100.txt random1_110x110.txt random1_121x121.txt random1_122x122.txt random1_128x128.txt random1_200x200.txt)

LINEAR_SOLVER_TESTS_A=(linear_2x2a.txt linear_3x3a.txt linear_5x5a.txt linear_25x25a.txt linear_50x50a.txt linear_100x100a.txt linear_150x150a.txt linear_200x200a.txt linear_250x250a.txt)
LINEAR_SOLVER_TESTS_b=(linear_2x1a.txt linear_3x1a.txt linear_5x1a.txt linear_25x1a.txt linear_50x1a.txt linear_100x1a.txt linear_150x1a.txt linear_200x1a.txt linear_250x1a.txt)

GJE_INVERSE_TESTS=(linear_2x2a.txt invert_3x3.txt invert_5x5.txt invert_10x10.txt invert_25x25.txt)

DETERMINANT_TESTS=(simple_2x2.txt simple_3x3.txt simple_4x4.txt simple_5x5.txt simple_6x6.txt simple_7x7.txt simple_8x8.txt simple_9x9.txt)


function run_test {
  #
  if [[ $# -eq 3 ]]; then
    $DIR/bin/matrix_ops $1 verify $2 $3 &> /dev/null
  else
    $DIR/bin/matrix_ops $1 verify $2 &> /dev/null
  fi

  return $?
}

function verify {
  if [[ $# -eq 3 ]]; then
    run_test $1 $TESTS/$2 $TESTS/$3
    if [ $? -ne 0 ]; then
      echo "$2 $3: FAILED"
      ALL_PASS=false
    else
      echo "$2 $3: PASSED"
    fi
  else
    run_test $1 $TESTS/$2
    if [ $? -ne 0 ]; then
      echo "$2: FAILED"
      ALL_PASS=false
    else
      echo "$2: PASSED"
    fi
  fi
}

function echo_header {
  printf "\n=================== %s ===================\n" "$1"
}

echo_header "MATRIX MULTIPLY"
for idx in "${!MM_A_TESTS[@]}"; do
  inputa=${MM_A_TESTS["$idx"]}
  inputb=${MM_B_TESTS["$idx"]}
  verify matmul $inputa $inputb
done

# LU Decomposition

echo_header "LU DECOMPOSITION"
for idx in "${!LU_DECOMP_TESTS[@]}"; do
  input=${LU_DECOMP_TESTS["$idx"]}
  verify lu_decomposition $input
done

echo_header "BLOCKED LU DECOMPOSITION"
for idx in "${!LU_DECOMP_TESTS[@]}"; do
  input=${LU_DECOMP_TESTS["$idx"]}
  verify lu_blockeddecomposition $input
done

echo_header "RANDOMIZED LU DECOMPOSITION"
for idx in "${!LU_SQDECOMP_TESTS[@]}"; do
  input=${LU_SQDECOMP_TESTS["$idx"]}
  verify lu_randomizeddecomposition $input
done
# GJE Inverse

echo_header "GJE Inverse"
for idx in "${!GJE_INVERSE_TESTS[@]}"; do
  input=${GJE_INVERSE_TESTS["$idx"]}
  verify GJE_inverse $input
done

# Linear System

echo_header "Steepest Descent Linear Solver"
for idx in {0..8}; do
  inputA=${LINEAR_SOLVER_TESTS_A["$idx"]}
  inputb=${LINEAR_SOLVER_TESTS_b["$idx"]}
  verify steepest_descent $inputA $inputb
done

echo_header "Conjugate Direction Linear Solver"
for idx in {0..8}; do
  inputA=${LINEAR_SOLVER_TESTS_A["$idx"]}
  inputb=${LINEAR_SOLVER_TESTS_b["$idx"]}
  verify conjugate_direction $inputA $inputb
done

echo_header "Inverse Linear Solver"
for idx in {0..3}; do
  inputA=${LINEAR_SOLVER_TESTS_A["$idx"]}
  inputb=${LINEAR_SOLVER_TESTS_b["$idx"]}
  verify inverse_linear_solver $inputA $inputb
done

# Determinant 

echo_header "Determinant"
for idx in "${!DETERMINANT_TESTS[@]}"; do
  input=${DETERMINANT_TESTS["$idx"]}
  verify determinant_lu $input
done

echo ""
if [ "$ALL_PASS" = true ]; then
  echo "All Tests Passed"
else
  echo "Some tests failed. See log for details"
fi
