
#include <iostream>

#include "common.h"
#include "matrix.h"
#include "cpu.h"

// Assumes that A->GetNumRows() >= A->GetNumCols()
void lu_decomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P)
{
  if (A->GetNumRows() < A->GetNumCols())
  {
    std::cerr << "lu_decomposition: matrix dimensions on A are ill formed for LU Decomposition" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (A->GetNumRows() != L->GetNumRows()
      || A->GetNumRows() != U->GetNumRows()
      || A->GetNumRows() != P->GetNumRows()
      || A->GetNumCols() != U->GetNumCols()
      || L->GetNumRows() != L->GetNumCols()
      || P->GetNumRows() != P->GetNumCols())
  {
    std::cerr << "lu_decomposition: matrix dimensions of inputs are mismatched" << std::endl;
  }
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

  P->ToIdentity(); // O(1)
  for (int i = 0; i < cols - 1; i++)
  {
    matrix_slicecolumn(U, column_slice, i); // O(1)
//    double slice[rows];
//    cudaMemcpy(slice, column_slice, rows * sizeof(double), cudaMemcpyDeviceToHost);
//    std::cout << "col " << i << " slice:\n{";
//    for (int j = 0; j < rows; ++j)
//    {
//      std::cout << " " << slice[j];
//    }
//    std::cout << " }" << std::endl;
    int idx;
    double max = reduce_absmaxidx(&column_slice[i], rows - i, &idx); // O(log(rows - i)) <= O(log(rows))
    idx = idx + i;
    std::cout << "col " << i << ", max: " << max << "@row " << idx;

    if (i != idx)
    {
      matrix_rowswap(P, i, idx); // O(1)
      matrix_rowswap(U, i, idx);
      matrix_subdiagonal_rowswap(L, i, idx);
    }

    // I reuse pointers in ways that don't match the names below
    // just to save on copies
    // Update U
    matrix_getelementarymatrix(U, L_intermediate, i);
    std::cout << "elementary matrix for col " << i << ": " << std::endl;
    matrix_print(L_intermediate);
    matrix_multiply(L_intermediate, U, U_intermediate);
    matrix_copy(U, U_intermediate);

    // Update L
    matrix_invertelementarymatrix(L_intermediate, P_intermediate, i);
    std::cout << "\ninverted elementary matrix:" << std::endl;
    matrix_print(P_intermediate);

    matrix_subdiagonal_writecolumn(L, P_intermediate, i);

    std::cout << "\nCurr U:" << std::endl;
    matrix_print(U);

    std::cout << "\nCurr P:" << std::endl;
    matrix_print(P);

    std::cout << "\nCurr L:" << std::endl;
    matrix_print(L);
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
