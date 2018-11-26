
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
    exit(EXIT_FAILURE);
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
//    std::cout << "col " << i << ", max: " << max << "@row " << idx;

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
//    std::cout << "elementary matrix for col " << i << ": " << std::endl;
//    matrix_print(L_intermediate);
    matrix_multiply(L_intermediate, U, U_intermediate);
    matrix_copy(U, U_intermediate);

    // Update L
    matrix_invertelementarymatrix(L_intermediate, P_intermediate, i);
//    std::cout << "\ninverted elementary matrix:" << std::endl;
//    matrix_print(P_intermediate);

    matrix_subdiagonal_writecolumn(L, P_intermediate, i);

//    std::cout << "\nCurr U:" << std::endl;
//    matrix_print(U);
//
//    std::cout << "\nCurr P:" << std::endl;
//    matrix_print(P);
//
//    std::cout << "\nCurr L:" << std::endl;
//    matrix_print(L);
  }
  
  delete U_intermediate;
  delete P_intermediate;
  delete P_acc;
  delete L_intermediate;
  cudaFree(column_slice);
}

void lu_blockeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P, int r)
{
  if (A->GetNumCols() < r)
  {
    r = A->GetNumCols();
  }
//  if (r < 2)
//  {
//    std::cerr << "lu_blockdecomposition: r width of submatrices must be greater than 1" << std::endl;
//    exit(EXIT_FAILURE);
//  }
  if (A->GetNumRows() < A->GetNumCols())
  {
    std::cerr << "lu_blockdecomposition: matrix dimensions on A are ill formed for LU Decomposition" << std::endl;
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
	Matrix *L_tl = new Matrix(r, r);
	Matrix *A_tr = new Matrix(r, A->GetNumCols() - r);
	Matrix *U_tr = new Matrix(r, A->GetNumCols() - r);
	Matrix *L_bl = new Matrix(A->GetNumRows() - r, r);
	Matrix *A_loop = new Matrix(A->GetNumRows(), A->GetNumCols());
	Matrix *A_loop_inter = new Matrix(A->GetNumRows(), A->GetNumCols());
	Matrix *L_loop = new Matrix(A->GetNumRows(), A->GetNumRows());
	Matrix *E_loop = new Matrix(A->GetNumRows(), A->GetNumRows());
	Matrix *E_loop_invert = new Matrix(A->GetNumRows(), A->GetNumRows());
//	Matrix *P_loop = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *A_copy = new Matrix(A->GetNumRows(), A->GetNumCols());
  double *column_slice;
  cudaMalloc((void **) &column_slice, sizeof(double) * A->GetNumRows());

  matrix_copy(A_loop, A);
  matrix_copy(A_copy, A);

//  P_loop->ToIdentity();

//  std::cout << "\nBeginning i loop" << std::endl;
  int o_r = r;
  for (int i = 0; i < A->GetNumCols(); i += o_r)
  {
    if (A->GetNumCols() <= i + r)
    {
      r = A->GetNumCols() - i;
      L_tl->ShrinkNumRows(r);
      L_tl->ShrinkNumCols(r);
      U_tl->ShrinkNumRows(r);
      U_tl->ShrinkNumCols(r);
      //std::cout << "\nNEW r: " << r << std::endl;
    }
		U_tr->ShrinkNumCols(A->GetNumCols() - (i + r));
    L_bl->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_tr->ShrinkNumCols(A->GetNumCols() - (i + r));

    L_loop->ToIdentity();

//    std::cout << "\nA_loop on i = " << i << std::endl;
//    matrix_print(A_loop);
//    std::cout << "\nA_copy" << std::endl;
//    matrix_print(A_copy);

		for (int j = 0; j < r; ++j)
		{
			matrix_slicecolumn(A_loop, column_slice, j); // O(1)
			int idx;
			double max = reduce_absmaxidx(&column_slice[j], (A_loop->GetNumRows() - j), &idx); // O(log(rows - i)) <= O(log(rows))
			idx = idx + j;

			if (j != idx)
			{
        matrix_rowswap(A_loop, j, idx);
        matrix_rowswap(A_copy, i + j, idx + i);
				matrix_rowswap(P, j + i, idx + i); // O(1)
				matrix_subdiagonal_rowswap(L_loop, j, idx);
        matrix_subdiagonal_rowswap(L, i + j, idx + i);
			}
    	matrix_getelementarymatrix(A_loop, E_loop, j);
//      std::cout << "\nAt first multiply" << std::endl;
//      std::cout << "\nE_loop" << std::endl;
//      matrix_print(E_loop);
//      std::cout << "\nA_loop" << std::endl;
//      matrix_print(A_loop); 
    	matrix_multiply(E_loop, A_loop, A_loop_inter);
    	matrix_copy(A_loop, A_loop_inter);
    
			matrix_invertelementarymatrix(E_loop, E_loop_invert, j);
    	matrix_subdiagonal_writecolumn(L_loop, E_loop_invert, j);
		}

    matrix_sliceblock(L_loop, L_tl, BlockLoc::UPPERLEFT);
    if (L_bl->GetNumRows() > 0)
    {
      matrix_sliceblock(L_loop, L_bl, BlockLoc::BOTTOMLEFT);
      matrix_writeblock(L, L_bl, r + i, i);
    }
    //matrix_sliceblock(A_loop, U_tr, BlockLoc::UPPERRIGHT);
    matrix_sliceblock(A_loop, U_tl, BlockLoc::UPPERLEFT);
//    std::cout << "\nA_copy" << std::endl;
//    matrix_print(A_copy);
//    std::cout << "\nA_tr" << std::endl;
//    matrix_print(A_tr);
    //matrix_writeblock(A_copy, A_loop, BlockLoc::BOTTOMRIGHT);
//		matrix_writeblock(P, P_loop, BlockLoc::BOTTOMRIGHT);

		A_loop->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_loop->ShrinkNumCols(A->GetNumCols() - (i + r));
		L_loop->ShrinkNumRows(A->GetNumRows() - (i + r));
		L_loop->ShrinkNumCols(A->GetNumRows() - (i + r));
		E_loop->ShrinkNumRows(A->GetNumRows() - (i + r));
		E_loop->ShrinkNumCols(A->GetNumRows() - (i + r));
//		P_loop->ShrinkNumRows(A->GetNumRows() - (i + r));
//		P_loop->ShrinkNumCols(A->GetNumCols() - (i + r));
		A_loop_inter->ShrinkNumRows(A->GetNumRows() - (i + r));
		A_loop_inter->ShrinkNumCols(A->GetNumCols() - (i + r));
		E_loop_invert->ShrinkNumRows(A->GetNumRows() - (i + r));
		E_loop_invert->ShrinkNumCols(A->GetNumRows() - (i + r));
		
//    matrix_sliceblock(P, P_loop, BlockLoc::BOTTOMRIGHT);
		matrix_writeblock(U, U_tl, i, i);
//    std::cout << "\nBefore L_tl update L_tl:" << std::endl;
//    matrix_print(L_tl);
//    std::cout << "\nBefore L_tl update Curr L:" << std::endl;
//    matrix_print(L);
		matrix_writeblock(L, L_tl, i, i);
//    std::cout << "\nAfter L_tl update Curr L:" << std::endl;
//    matrix_print(L);
    
    if (A_tr->GetNumCols() > 0)
    {
      matrix_sliceblock(A_copy, A_loop, BlockLoc::BOTTOMRIGHT);
		  matrix_sliceblock(A_copy, A_tr, i, i + r);
      GJE_inverse(L_tl);
//      std::cout << "\ninverse of L_tl" << std::endl;
//      matrix_print(L_tl);
//      std::cout << "\nA_tr" << std::endl;
//      matrix_print(A_tr);
      matrix_multiply(L_tl, A_tr, U_tr);
     
//      std::cout << "\nobtained U_tr" << std::endl;
//      matrix_print(U_tr);
//      std::cout << "\nL_bl" << std::endl;
//      matrix_print(L_bl);
      matrix_multiply(L_bl, U_tr, A_loop_inter);
      matrix_subtract(A_loop, A_loop_inter, A_loop);
//      std::cout << "\nnew A_loop" << std::endl;
//      matrix_print(A_loop);
//      std::cout << "\nBefore write A_copy" << std::endl;
//      matrix_print(A_copy);
      matrix_writeblock(A_copy, A_loop, BlockLoc::BOTTOMRIGHT);
//      std::cout << "\nAfter A_copy" << std::endl;
//      matrix_print(A_copy);
      matrix_writeblock(U, U_tr, i, r + i);
    }
    
//    std::cout << "\nCurr U:" << std::endl;
//    matrix_print(U);
//
//    std::cout << "\nCurr P:" << std::endl;
//    matrix_print(P);
//
//    std::cout << "\nCurr L:" << std::endl;
//    matrix_print(L);
  }
	
	delete U_tl;
	delete L_tl;
	delete A_tr;
	delete U_tr;
	delete L_bl;
  delete A_copy;
//	delete P_loop;
	delete A_loop;
	delete A_loop_inter;
	delete L_loop;
	delete E_loop;
	delete E_loop_invert;
  delete A_copy;
  cudaFree(column_slice);
}
