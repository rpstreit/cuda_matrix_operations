
#include <cstdarg>
#include <iostream>

#include "tests.h"
#include "common.h"
#include "cpu.h"
#include "matrix.h"
#include "lu_decomposition.h"
#include "linearSysSolver.h"

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
  std::cout << "\nAB = " << std::endl;
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
  if (A->GetNumCols() != B->GetNumRows())
  {
    std::cerr << "error: num cols in input 1 does not equal num rows of input 2" << std::endl;
    exit(EXIT_FAILURE);
  }

  Matrix *result_cpu = new Matrix(A->GetNumRows(), B->GetNumCols());
  Matrix *result_gpu = new Matrix(A->GetNumRows(), B->GetNumCols());
  
  matrix_multiply_cpu(A, B, result_cpu);
  matrix_multiply(A, B, result_gpu);

  cudaDeviceSynchronize();
  int result = matrix_equals(result_cpu, result_gpu, ERROR) ? 0 : 1;

  cudaDeviceSynchronize();
  delete result_cpu;
  delete result_gpu;

  return result;
}

int lu_decomposition_run(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *L = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  lu_decomposition(A, L, U, P);

  std::cout << "\nP =" << std::endl;
  matrix_print(P);
  std::cout << "\nL =" << std::endl;
  matrix_print(L);
  std::cout << "\nU =" << std::endl;
  matrix_print(U);

  delete P;
  delete L;
  delete U;
  delete left;
  delete right;

  return 0;
}

int lu_decomposition_verify(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *L = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  lu_decomposition(A, L, U, P);

  matrix_multiply_cpu(P, A, left);
  matrix_multiply_cpu(L, U, right);

  std::cout << "\nPA = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  int result = matrix_equals(left, right, ERROR) ? 0 : 1;

  delete P;
  delete L;
  delete U;
  delete left;
  delete right;

  return result;
}

int linear_descent_run(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output = steepestDescent(A_operator, b_operator);
  matrix_print(output);

  delete A_operator;
  delete b_operator;
  delete output;

  return 0;
}

int conjugate_direction_run(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }
  
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output = conjugateDirection(A_operator, b_operator);
  matrix_print(output);

  delete A_operator;
  delete b_operator;
  delete output;

  return 0;
}

int determinant_recur_run(int argc, Matrix **argv)
{
  if(argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);

  int determinant = determinant_recur(A_operator);
  std::cout << determinant << std::endl;

  delete A_operator;
}

int linear_solve_verify(int argc, Matrix **argv)
{
  return 0;
}

int determinant_verify(int argc, Matrix **argv)
{
  return 0;
}
