
// Please create two seperate functions for each
// operation. One that simply dumps results to 
// command line, and one that does some sort of
// verification (doing the same operation with
// lapack for example and comparing the results).
//
// For both please return 0 on success. This is 
// more important for the verifier function


// Here is a constant to account for floating
// precision loss when checking for equality

#ifndef TESTS_H
#define TESTS_H

#include "matrix.h"

#define ERROR 1e-10

int matmul_run(int argc, Matrix **argv);

int matmul_verify(int argc, Matrix **argv);

int lu_decomposition_run(int argc, Matrix **argv);

int lu_decomposition_verify(int argc, Matrix **argv);

int steepest_descent_run(int argc, Matrix **argv);

int conjugate_direction_run(int argc, Matrix **argv);

int inverse_linear_run(int argc, Matrix **argv); 

int steepest_descent_verify(int argc, Matrix **argv);

int conjugate_direction_verify(int argc, Matrix **argv);

int determinant_recur_run(int argc, Matrix **argv);

int determinant_lu_run(int argc, Matrix **argv);

int determinant_verify(int argc, Matrix **argv);

int GJE_inverse_run(int argc, Matrix **argv);

int inverse_verify(int argc, Matrix **argv);

int inverse_linear_verify (int argc, Matrix **argv);

#endif
