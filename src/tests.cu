
#include <cstdarg>
#include <iostream>

#include "tests.h"
#include "common.h"
#include "cpu.h"
#include "matrix.h"

int matmul_run(int argc, Matrix **argv) 
{
  if (argc != 2)
  {
    std::cerr << "error: matmul requires 2 arguments" << std::endl;
    exit(EXIT_FAILURE);
  }

  Matrix *A = argv[0];
  Matrix *B = argv[1];
  if (A->GetNumCols() != B->GetNumRows())
  {
    std::cerr << "error: num cols in input 1 does not equal num rows of input 2" << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix *result = new Matrix(A->GetNumRows(), B->GetNumCols());
  
  matrix_multiply(A, B, result);
  std::cout << "AB = " << std::endl;
  matrix_print(result);

  delete result;
  return 0;
}

int matmul_verify(int argc, Matrix **argv)
{
  if (argc != 2)
  {
    std::cerr << "error: matmul requires 2 arguments" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  Matrix *A = argv[0];
  Matrix *B = argv[1];
  Matrix *result_cpu = new Matrix(A->GetNumRows(), B->GetNumCols());
  Matrix *result_gpu = new Matrix(A->GetNumRows(), B->GetNumCols());
  
  matrix_multiply_cpu(A, B, result_cpu);
  matrix_multiply(A, B, result_gpu);

  return matrix_equals(result_cpu, result_gpu, ERROR) ? 0 : 1;
}
