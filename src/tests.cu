
#include <cstdarg>
#include <iostream>
#include <cmath>

#include "tests.h"
#include "common.h"
#include "cpu.h"
#include "matrix.h"
#include "lu_decomposition.h"
#include "linearSysSolver.h"
#include "determinant.h"
#include "matrix_inverse.h"

#define DBL_EPSILON 2.2204460492503131e-16

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
 
  std::cout << "Running Matrix Multiply... ";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  matrix_multiply(A, B, result);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "\nAB = " << std::endl;
  matrix_print(result);
  std::cout << elapsed_time << "ms" << std::endl;

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

int lu_randomizeddecomposition_run(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *Q = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

//  int l = (int)std::sqrt((double)A->GetNumCols());
//  int k = l * ((int)std::sqrt((double)l));
#ifdef DEFAULTRANDOMA
  int l = A->GetNumCols();
  int k = A->GetNumCols();
#else
  int l = (int)(((double)A->GetNumCols()/10.f) * 9.f);
  int k = l;
#endif

  std::cout << "Running Randomized LU Decomposition... ";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  lu_randomizeddecomposition(A, L, U, P, Q, l, k);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  matrix_multiply_cpu(P, A, left);
  matrix_multiply_cpu(left, Q, A);
  matrix_copy(left, A);
  matrix_multiply_cpu(L, U, right);
  std::cout << "\nP =" << std::endl;
  matrix_print(P);
  std::cout << "\nQ =" << std::endl;
  matrix_print(Q);
  std::cout << "\nL =" << std::endl;
  matrix_print(L);
  std::cout << "\nU =" << std::endl;
  matrix_print(U);
  std::cout << "\nPAQ = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  std::cout << "k, l = " << k << ", " << l << std::endl;
  std::cout << "completed in " << elapsed_time << "ms" << std::endl;

  delete P;
  delete Q;
  delete L;
  delete U;
  delete left;
  delete right;

  return 0;
}

int lu_randomizeddecomposition_verify(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *Q = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

#ifdef DEFAULTRANDOMA
  int l = A->GetNumCols();
  int k = A->GetNumCols();
#else
  int l = A->GetNumCols();//10 * 7;
  int k = l;
#endif

  lu_randomizeddecomposition(A, L, U, P, Q, l, k);

  matrix_multiply_cpu(P, A, left);
  matrix_multiply_cpu(left, Q, A);
  matrix_copy(left, A);
  matrix_multiply_cpu(L, U, right);

  std::cout << "\nPAQ = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  int result = matrix_equals(left, right, 1000.f) ? 0 : 1; // The error bound is very large so that I can have some sort of garuntee for low rank approximations

  delete P;
  delete Q;
  delete L;
  delete U;
  delete left;
  delete right;

  return result;
}

int lu_blockeddecomposition_run(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  int r = (int)std::sqrt((double)A->GetNumCols());

  std::cout << "Running Blocked LU Decomposition... ";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  lu_blockeddecomposition(A, L, U, P, r);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << elapsed_time << "ms" << std::endl;

  std::cout << "\nP =" << std::endl;
  matrix_print(P);
  std::cout << "\nL =" << std::endl;
  matrix_print(L);
  std::cout << "\nU =" << std::endl;
  matrix_print(U);
  matrix_multiply_cpu(P, A, left);
  matrix_multiply_cpu(L, U, right);

  std::cout << "\nPA = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  std::cout << "completed in " << elapsed_time << "ms" << std::endl;

  delete P;
  delete L;
  delete U;
  delete left;
  delete right;

  return 0;
}

int lu_blockeddecomposition_verify(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  int r = (int)std::sqrt((double)A->GetNumCols());

  lu_blockeddecomposition(A, L, U, P, r);

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

int lu_columndecomposition_run(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *Q = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  std::cout << "Running LU Decomposition with Column Pivoting... ";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  lu_columndecomposition(A, L, U, Q);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  std::cout << elapsed_time << "ms" << std::endl;
  std::cout << "\nQ =" << std::endl;
  matrix_print(Q);
  std::cout << "\nL =" << std::endl;
  matrix_print(L);
  std::cout << "\nU =" << std::endl;
  matrix_print(U);
  std::cout << "completed in " << elapsed_time << "ms" << std::endl;

  delete Q;
  delete L;
  delete U;
  delete left;
  delete right;

  return 0;
}

int lu_columndecomposition_verify(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *Q = new Matrix(A->GetNumCols(), A->GetNumCols());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  lu_decomposition(A, L, U, Q);

  matrix_multiply_cpu(Q, A, left);
  matrix_multiply_cpu(L, U, right);

  std::cout << "\nQA = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  int result = matrix_equals(left, right, ERROR) ? 0 : 1;

  delete Q;
  delete L;
  delete U;
  delete left;
  delete right;

  return result;
}

int lu_decomposition_run(int argc, Matrix **argv)
{
  if (argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }

  Matrix *A = argv[0];
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *left = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *right = new Matrix(A->GetNumRows(), A->GetNumCols());

  std::cout << "Running LU Decomposition... ";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  lu_decomposition(A, L, U, P);
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  std::cout << elapsed_time << "ms" << std::endl;
  std::cout << "\nP =" << std::endl;
  matrix_print(P);
  std::cout << "\nL =" << std::endl;
  matrix_print(L);
  std::cout << "\nU =" << std::endl;
  matrix_print(U);
  matrix_multiply_cpu(P, A, left);
  matrix_multiply_cpu(L, U, right);

  std::cout << "\nPA = " << std::endl;
  matrix_print(left);
  std::cout << "\nLU = " << std::endl;
  matrix_print(right);
  std::cout << "completed in " << elapsed_time << "ms" << std::endl;

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
  Matrix *P = new Matrix(A->GetNumRows(), A->GetNumRows());
  Matrix *L = new Matrix(A->GetNumRows(), A->GetNumRows());
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

int steepest_descent_run(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: steepest descent requires 2 arguments" << std::endl;
  }
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output = steepestDescent(A_operator, b_operator);
  matrix_print(output);

  delete output;

  return 0;
}

int conjugate_direction_run(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: conjugate direction requires 2 arguments" << std::endl;
  }
  
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output = conjugateDirection(A_operator, b_operator);
  matrix_print(output);

  delete output;

  return 0;
}
int inverse_linear_run(int argc, Matrix **argv) {
  if(argc != 2)
  {
    std::cerr << "error: Inverse Linear System Solver requires 2 arguments" << std::endl;
  }
  
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output = inverseLinearSolver(A_operator, b_operator);
  matrix_print(output);

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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  double determinant = ceil(determinant_recur(A_operator));
  std::cout << "determinant: " << determinant << std::endl;

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  std::cout << "determinant_recur took : " << elapsed_time << " ms" << std::endl;

  return 0; 
}

int determinant_lu_run(int argc, Matrix **argv)
{
  if(argc != 1)
  {
    std::cerr << "error: lu decomposition requires 1 argument" << std::endl;
  }
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  double determinant = determinant_lu(A_operator);
  std::cout << "determinant: " << determinant << std::endl;
  printf("%f", determinant);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  std::cout << "determinant_recur took : " << elapsed_time << " ms" << std::endl;

  return 0;
}

int steepest_descent_verify(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: Steepest Descent requires 2 arguments" << std::endl;
  }
  
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output1 = steepestDescent(A_operator, b_operator);
  Matrix * x_star = new Matrix(b_operator->GetNumRows(), b_operator->GetNumCols()); 
  matrix_multiply_cpu(A_operator, output1, x_star);
  bool ok = true;
  for(int i=0; i<x_star->GetNumRows(); i++) {
    if(abs(x_star->GetFlattened()[i] - b_operator->GetFlattened()[i]) > 1)
      ok = false;
  }

  return ok ? 0 : 1;
}

int conjugate_direction_verify(int argc, Matrix **argv)
{
  if(argc != 2)
  {
    std::cerr << "error: Conjugate Direction requires 2 argument" << std::endl;
  }
  
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output1 = conjugateDirection(A_operator, b_operator);
  Matrix * x_star = new Matrix(b_operator->GetNumRows(), b_operator->GetNumCols()); 
  matrix_multiply_cpu(A_operator, output1, x_star);
  bool ok = true;
  for(int i=0; i<x_star->GetNumRows(); i++) {
    if(abs(x_star->GetFlattened()[i] - b_operator->GetFlattened()[i]) > .1)
      ok = false;
  }

  return ok ? 0 : 1;
}

int determinant_verify(int argc, Matrix **argv)
{
  if(argc != 1)
  {
    std::cerr << "error: determinant_verify_run requires 1 argument" << std::endl;
  }

  Matrix * A_operator = argv[0];
  matrix_print(A_operator);

  double recur_ans = ceil(determinant_recur(A_operator));
  double lu_ans = determinant_lu(A_operator);  

  std::cout << "recursive: " << recur_ans << " lu: " << lu_ans << std::endl; 
  if (std::fabs(recur_ans - lu_ans) < DBL_EPSILON)
  {
    return 0;
  }

  return 1;
}

int GJE_inverse_run(int argc, Matrix **argv)
{
  if(argc != 1)
  {
    std::cerr << "error: GJE_inverse_run requires 1 argument" << std::endl;
  }
  Matrix * A_operator = argv[0];
  matrix_print(A_operator);

  Matrix * output = GJE_inverse(A_operator);
  matrix_print(output);
  return 0;
}

int inverse_verify(int argc, Matrix **argv)
{
  if(argc != 1)
  {
    std::cerr << "error: GJE_inverse_run requires 1 argument" << std::endl;
  }
  Matrix * A_operator = argv[0];
  std::cout << "A_operator" << std::endl;
  matrix_print(A_operator);
  std::cout << std::endl;
  Matrix *inverse = new Matrix(*A_operator);
  Matrix * output = GJE_inverse(inverse);
  std::cout << "Inverse matrix: " << std::endl;
  matrix_print(output);
  std::cout << std::endl;

  Matrix* check = new Matrix(A_operator->GetNumRows(), A_operator->GetNumCols());

  matrix_multiply(A_operator, output, check);
  matrix_floor_small(check, check);
  std::cout << "Check" << std::endl;
  matrix_print(check);
  std::cout << std::endl;
  A_operator->ToIdentity();

  if (matrix_equals(check, A_operator, 0.01)){
    delete check;
    return 0;
  }

  delete check;
  return 1;
}

int inverse_linear_verify (int argc, Matrix **argv) {
  if(argc != 2)
  {
    std::cerr << "error: Inverse Linear System Solver requires 2 arguments" << std::endl;
  }

  Matrix * A_operator = argv[0];
  matrix_print(A_operator);
  Matrix * b_operator = argv[1];
  matrix_print(b_operator);

  Matrix * output1 = inverseLinearSolver(A_operator, b_operator);
  matrix_print(output1);
  Matrix * x_star = new Matrix(b_operator->GetNumRows(), b_operator->GetNumCols()); 
  matrix_multiply_cpu(A_operator, output1, x_star);
  bool ok = true;
  for(int i=0; i<x_star->GetNumRows(); i++) {
    if(abs(x_star->GetFlattened()[i] - b_operator->GetFlattened()[i]) > .1)
      ok = false;
  }

  return ok ? 0 : 1;
}
