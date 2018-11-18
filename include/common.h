
class enum Reduction
{
  ADD, MIN, MAX, MUL
}

// reduce
//
// Computes a specified reduction operation in parallel
//
// Inputs: flattened data array, length of such array, reduction
// operation wished to be performed
// Outputs: double result of reduction
double reduce(double *data, int length, Reduction op_type);

// matrix_transpose
//
// Computes matrix transpose in parallel
//
// Inputs: Managed matrix pointer, pre allocated managed matrix 
// result pointer. Note: ensure that for mxn matrix mat result is 
// nxm
// Outputs: Resulting transpose in result
void matrix_transpose(Matrix *mat, Matrix *result)
