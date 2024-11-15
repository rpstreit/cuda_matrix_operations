
#ifndef COMMON_H
#define COMMON_H

#include "matrix.h"

#define THREADS_PER_BLOCK 256

enum Reduction
{
  ADD, MIN, MAX, MUL
};

enum BlockLoc
{
  UPPERLEFT, UPPERRIGHT, BOTTOMLEFT, BOTTOMRIGHT 
};

// reduce
//
// Computes a specified reduction operation in parallel
//
// Inputs: flattened data array, length of such array, reduction
// operation wished to be performed
// Outputs: double result of reduction
double reduce(double *data, int length, Reduction op_type);

double reduce_absmaxidx(double *data, int length, int *idx);

void matrix_normrandomize(Matrix *A);

// matrix_transpose
//
// Computes matrix transpose in parallel
//
// Inputs: Managed matrix pointer, pre allocated managed matrix 
// result pointer. Note: ensure that for mxn matrix mat result is 
// nxm
// Outputs: Resulting transpose in result
void matrix_transpose(Matrix *mat, Matrix *result);

// matrix_multiply
//
// Computes a matrix matrix multiplication in parallel
//
// Inputs: Managed matrices A and B pointers, pre allocated matrix
// result pointer. Note: ensure that for mxl * lxn matrix multiply
// result is mxn
// Outpus: Resulting multiply in result
void matrix_multiply(Matrix *A, Matrix *B, Matrix *result);

void matrix_writeblock(Matrix *dest, Matrix *src_block, BlockLoc loc);

void matrix_writeblock(Matrix *dest, Matrix *src, int tlx, int tly);

void matrix_sliceblock(Matrix *src, Matrix *dest, BlockLoc loc);

void matrix_sliceblock(Matrix *src, Matrix *dest, int tlx, int tly);

void matrix_slicecolumn(Matrix *A, double *slice, int col_idx); 

void matrix_copy(Matrix *dest, Matrix *src);

void matrix_rowswap(Matrix *A, int row1, int row2);

void matrix_colswap(Matrix *A, int col1, int col2);

void matrix_subdiagonal_rowswap(Matrix *A, int row1, int row2);

void matrix_add(Matrix *A, Matrix *B, Matrix *C);

void matrix_subtract(Matrix *A, Matrix *B, Matrix *C);

void matrix_multiply_scalar(Matrix *output, Matrix *input, double scale);

void matrix_getelementarymatrix(Matrix *A, Matrix *result, int col);

void matrix_invertelementarymatrix(Matrix *A, Matrix *result, int col);

bool matrix_equals(Matrix *A, Matrix *B, double error);

void matrix_floor_small(Matrix* output, Matrix *input);

double dot_product(Matrix *vec1, Matrix *vec2);

void matrix_print(Matrix *A);

double norm(Matrix *vec);

void matrix_subdiagonal_writecolumn(Matrix *dest, Matrix *src, int col);
#endif
