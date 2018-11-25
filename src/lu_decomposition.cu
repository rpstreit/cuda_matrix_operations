
#include <iostream>

#include "common.h"
#include "matrix.h"
#include "matrix_inverse.h"
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

void lu_blockeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P, int r)
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

  P->ToIdentity();
  L->ToIdentity();
  U->ToZeroes();

	Matrix *U_tl = new Matrix(r, r);
	Matrix *U_tl_inter = new Matrix(r, r);	
	Matrix *L_tl = new Matrix(r, r);
	Matrix *L_tl_inter = new Matrix(r, r);	
	Matrix *P_tl = new Matrix(r, r);
	Matrix *P_tl_inter = new Matrix(r, r);	
	Matrix *A_tr = new Matrix(r, A->GetNumCols() - r);
	Matrix *U_tr = new Matrix(r, A->GetNumCols() - r);
	Matrix *U_tr_inter = new Matrix(r, A->GetNumCols() - r);
	Matrix *A_bl = new Matrix(A->GetNumRows() - r, r);
	Matrix *L_bl = new Matrix(A->GetNumRows() - r, r);
	Matrix *L_bl_inter = new Matrix(A->GetNumRows() - r, r);
	Matrix *A_br = new Matrix(A->GetNumRows() - r, A->GetNumCols( ) - r);
	Matrix *A_br_inter = new Matrix(A->GetNumRows() - r, A->GetNumCols() - r);
  double *column_slice;
  cudaMalloc((void **) &column_slice, sizeof(double) * r);

  for (int i = 0; i < A->GetNumCols(); i += r)
  {
		if (i > A->GetNumCols() - r)
		{
			r = A->GetNumCols() - i;
			U_tl->ShrinkNumCols(r);
			U_tl->ShrinkNumRows(r);
			U_tl_inter->ShrinkNumCols(r);
			U_tl_inter->ShrinkNumRows(r);
			P_tl->ShrinkNumCols(r);
			P_tl->ShrinkNumRows(r);
			P_tl_inter->ShrinkNumCols(r);
			P_tl_inter->ShrinkNumRows(r);
			L_tl->ShrinkNumCols(r);
			L_tl->ShrinkNumRows(r);
			L_tl_inter->ShrinkNumCols(r);
			L_tl_inter->ShrinkNumRows(r);

			U_tr->ShrinkNumRows(r);
			U_tr_inter->ShrinkNumRows(r);
			A_tr->ShrinkNumRows(r);
			L_bl->ShrinkNumCols(r);
			L_bl_inter->ShrinkNumCols(r);
			A_bl->ShrinkNumCols(r);
		}
		A_tr->ShrinkNumCols(A->GetNumCols() - (i + r));
		U_tr->ShrinkNumCols(A->GetNumCols() - (i + r));
		U_tr_inter->ShrinkNumCols(A->GetNumCols() - (i + r));
		A_bl->ShrinkNumRows(A->GetNumRows() - (i + r));
		L_bl->ShrinkNumRows(A->GetNumRows() - (i + r));
		L_bl_inter->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_br->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_br->ShrinkNumCols(A->GetNumCols() - (i + r));
		A_br_inter->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_br_inter->ShrinkNumCols(A->GetNumCols() - (i + r));
			
		matrix_sliceblock(A, U_tl, i, i);
		matrix_sliceblock(A, A_bl, i + r, i);
		matrix_sliceblock(A, A_tr, i, i + r);

		for (int j = 0; j < r; ++j)
		{
			matrix_slicecolumn(U_tl, column_slice, j); // O(1)
			int idx;
			double max = reduce_absmaxidx(&column_slice[i], r - j, &idx); // O(log(rows - i)) <= O(log(rows))
			idx = idx + j;

			if (j != idx)
			{
				matrix_rowswap(P_tl, j, idx); // O(1)
				matrix_rowswap(U_tl, j, idx);
				matrix_subdiagonal_rowswap(L_tl, j, idx);
			}
    	matrix_getelementarymatrix(U_tl, L_tl_inter, j);
    	matrix_multiply(U_tl_inter, U_tl, U_tl_inter);
    	matrix_copy(U_tl, U_tl_inter);
    
			matrix_invertelementarymatrix(L_tl_inter, P_tl_inter, j);
    	matrix_subdiagonal_writecolumn(L_tl, P_tl_inter, j);
		}

		matrix_copy(L_tl_inter, L_tl);
		matrix_copy(U_tl_inter, U_tl);
		GJE_inverse(U_tl_inter);
		GJE_inverse(L_tl_inter);
		matrix_multiply(A_bl, U_tl_inter, L_bl);
		matrix_multiply(L_tl_inter, A_tr, U_tr);
		
		matrix_multiply(L_bl, U_tr, A_br_inter);
		matrix_subtract(A_br, A_br_inter, A_br);

		matrix_writeblock(A, A_br, BlockLoc::BOTTOMRIGHT);
		matrix_writeblock(P, P_tl, i, i);
		matrix_writeblock(U, U_tl, i, i);
		matrix_writeblock(L, L_tl, i, i);
		matrix_writeblock(U, U_tr, i, r + i);
		matrix_writeblock(L, L_bl, r + i, i);
  }
	
	delete U_tl;
	delete U_tl_inter;	
	delete L_tl;
	delete L_tl_inter;	
	delete P_tl;
	delete P_tl_inter;	
	delete A_tr;
	delete U_tr;
	delete U_tr_inter;
	delete A_bl;
	delete L_bl;
	delete L_bl_inter;
	delete A_br;
	delete A_br_inter;
  cudaFree(column_slice);
}
