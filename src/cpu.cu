
#include "matrix.h"

void matrix_multiply_cpu(Matrix *A, Matrix *B, Matrix *result)
{
  result->ToZeroes();

  for (int r = 0; r < A->GetNumRows(); r++) 
  {
    for (int c = 0; c < B->GetNumCols(); c++)
    {
      for (int in = 0; in < A->GetNumCols(); in++)
      {
          (*result)[r][c] += (*A)[r][in] * (*B)[in][c];
      }
    }
  }
}
