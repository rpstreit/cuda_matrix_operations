
#ifndef COMMON_H
#define COMMON_H

#define THREADS_PER_BLOCK 256

class enum Reduction
{
  ADD, MIN, MAX, MUL
}

class enum BlockLoc
{
  UPPERLEFT, UPPERRIGHT, BOTTOMLEFT, BOTTOMRIGHT 
}

// reduce
//
// Computes a specified reduction operation in parallel
//
// Inputs: flattened data array, length of such array, reduction
// operation wished to be performed
// Outputs: double result of reduction
double reduce(double *data, int length, Reduction op_type);

std::tuple<int, double> reduce_maxidx(double *data, int length);

// matrix_transpose
//
// Computes matrix transpose in parallel
//
// Inputs: Managed matrix pointer, pre allocated managed matrix 
// result pointer. Note: ensure that for mxn matrix mat result is 
// nxm
// Outputs: Resulting transpose in result
void matrix_transpose(Matrix *mat, Matrix *result)

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

void matrix_columnslice(Matrix *A, double *slice, int col_idx);

#endif
