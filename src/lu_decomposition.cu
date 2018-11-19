
#include <tuple>

#include "common.h"
#include "matrix.h"

// Assumes that A->GetNumRows() >= A->GetNumCols()
void lu_decomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P)
{
  double *column_slice;
  P->ToIdentity();
  L->ToIdentity();
  matrix_copy(U, A);
  int rows = A->GetNumRows();
  int cols = A->GetNumCols();

  Matrix *U_intermediate = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *P_intermediate = new Matrix(P->GetNumRows(), P->GetNumCols());
  Matrix *P_acc = new Matrix(P->GetNumRows(), P->GetNumCols());
  Matrix *L_intermediate = new Matrix(L->GetNumRows(), L->GetNumCols());
  cudaMalloc((void **) &column_slice, sizeof(double) * rows);

  for (int i = 0; i < cols - 1; i++)
  {
    P_intermediate->ToIdentity(); // O(1)
    matrix_slicecolumn(A, column_slice, i); // O(1)
    std::tuple<int, double> max_idx = reduce_maxidx(column_slice + i, rows - i); // O(log(rows - i)) <= O(log(rows))
    int idx = std::get<0>(max_idx);
    double max = std::get<1>(max_idx);

    if (i != idx)
    {
      matrix_rowswap(P_intermediate, i, idx); // O(1)
    }

    // Update U
    matrix_multiply(P_intermediate, U, U_intermediate);
    matrix_getelementarymatrix(U_intermediate, L_intermediate, i);
    matrix_multiply(L_intermediate, U_intermediate, U);

    // Update P
    matrix_multiply(P_intermediate, P, P_acc);
    matrix_copy(P, P_acc);

    // Update L
    matrix_invertelementarymatrix(L_intermediate, P_intermediate, i); // O(1) baby
    matrix_multiply(P_intermediate, L, L_intermediate); // Something logarithmic
    matrix_copy(L, L_intermediate); // O(1)
  }
  
  delete U_intermediate;
  delete P_intermediate;
  delete P_acc;
  delete L_intermediate;
  cudaFree(column_slice);
}

void lu_blockeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P)
{

}
