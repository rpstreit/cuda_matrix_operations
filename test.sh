#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0 ]}" )" && pwd )"
TESTS=${DIR}/tests
ALL_PASS=true

LU_DECOMP_TESTS=(simple_3x3.txt simple_3x2.txt simple_4x4.txt simple_5x5.txt simple_8x8.txt simple_9x9.txt random1_7x4.txt random1_10x10.txt random1_17x17.txt random1_18x18.txt random1_19x19.txt random1_20x20.txt random1_27x27.txt random1_32x32.txt random1_37x37.txt random1_38x38.txt random1_39x39.txt random1_48x48.txt random1_50x50.txt random1_96x47.txt random1_98x98.txt random1_99x99.txt random1_100x100.txt random1_110x110.txt random1_121x121.txt random1_122x122.txt random1_128x128.txt random1_200x200.txt)

make

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
      echo "$2 $3: FAILED"
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

echo ""
if [ "$ALL_PASS" = true ]; then
  echo "All Tests Passed"
else
  echo "Some tests failed. See log for details"
fi