
#include <iostream>

#include "common.h"
#include "matrix.h"
#include "matrix_inverse.h"
#include "cpu.h"

__global__ void kfix_diagonal(Matrix *L);

// Assumes that A->GetNumRows() >= A->GetNumCols()
void lu_decomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P)
{
  if (A->GetNumRows() < A->GetNumCols())
  {
    matrix_print(A);
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

void lu_columndecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *Q)
{
//  if (A->GetNumRows() < A->GetNumCols())
//  {
//    matrix_print(A);
//    std::cerr << "lu_decomposition: matrix dimensions on A are ill formed for LU Decomposition" << std::endl;
//    exit(EXIT_FAILURE);
//  }
  if (A->GetNumRows() != L->GetNumRows()
      || A->GetNumRows() != U->GetNumRows()
      || A->GetNumCols() != Q->GetNumRows()
      || A->GetNumCols() != U->GetNumCols()
//      || L->GetNumRows() != L->GetNumCols()
      || Q->GetNumRows() != Q->GetNumCols())
  {
    std::cerr << "lu_decomposition: matrix dimensions of inputs are mismatched" << std::endl;
    if (A->GetNumRows() != L->GetNumRows())
    {
      std::cerr << "A->GetNumRows() != L->GetNumRows()" << std::endl;
    }
    if (A->GetNumRows() != U->GetNumRows())
    {
      std::cerr << "A->GetNumRows() != U->GetNumRows()" << std::endl;
    }
    if (A->GetNumCols() != Q->GetNumRows())
    {
      std::cerr << "A->GetNumCols() != Q->GetNumRows()" << std::endl;
    }
    if (A->GetNumCols() != U->GetNumCols())
    {
      std::cerr << "A->GetNumCols() != U->GetNumCols()" << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  Q->ToIdentity();
  L->ToIdentity();
  matrix_copy(U, A);
  int rows = A->GetNumRows();
  int cols = A->GetNumCols();

  Matrix *U_intermediate = new Matrix(A->GetNumRows(), A->GetNumCols());
  Matrix *E = new Matrix(L->GetNumRows(), L->GetNumRows());
  Matrix *L_intermediate = new Matrix(L->GetNumRows(), L->GetNumCols());

  for (int i = 0; i < cols - 1 && i < rows - 1; i++)
  {
    int idx;
    double max = reduce_absmaxidx(&(U->GetFlattened()[i * U->GetNumCols() + i]), cols - i, &idx); // O(log(rows - i)) <= O(log(rows))
    idx = idx + i;

//    std::cout << "MAX idx: " << idx << std::endl;
    if (i != idx)
    {
      matrix_colswap(Q, i, idx); // O(1)
      matrix_colswap(U, i, idx);
//      matrix_subdiagonal_colswap(L, i, idx);
    }

    // I reuse pointers in ways that don't match the names below
    // just to save on copies
    // Update U
    matrix_getelementarymatrix(U, E, i);
    matrix_multiply(E, U, U_intermediate);
    matrix_copy(U, U_intermediate);

    // Update L
    matrix_invertelementarymatrix(E, L_intermediate, i);
    matrix_subdiagonal_writecolumn(L, L_intermediate, i);
  }
  
  delete U_intermediate;
  delete L_intermediate;
}
//void lu_decomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P)
void lu_randomizeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P, Matrix *Q, int l, int k)
{
  if (k == 1)
  {
    std::cerr << "keeping only 1 singular value ain't gonna work" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (A->GetNumRows() != A->GetNumCols())
  {
    matrix_print(A);
    std::cerr << "lu_decomposition: matrix dimensions on A are ill formed for LU Decomposition with complete pivoting" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (A->GetNumRows() != L->GetNumRows()
      || A->GetNumRows() != U->GetNumRows()
      || A->GetNumCols() != Q->GetNumRows()
      || A->GetNumRows() != P->GetNumCols()
      || A->GetNumCols() != U->GetNumCols()
      || L->GetNumRows() != L->GetNumCols()
      || Q->GetNumRows() != Q->GetNumCols()
      || P->GetNumCols() != P->GetNumRows())
  {
    std::cerr << "lu_decomposition: matrix dimensions of inputs are mismatched" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (l < k)
  {
    std::cerr << "l must be greater than k" << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix *G = new Matrix(A->GetNumCols(), l);
  Matrix *Y = new Matrix(A->GetNumRows(), l);
  
  matrix_normrandomize(G);
//  std::cout << "G: " << std::endl;
//  matrix_print(G);
  matrix_multiply(A, G, Y);
//  std::cout << "Y: " << std::endl;
//  matrix_print(Y);
  delete G;
  
  Matrix *L_y = new Matrix(Y->GetNumRows(), Y->GetNumRows());
  Matrix *U_y = new Matrix(Y->GetNumRows(), Y->GetNumCols());
//  std::cout << "L_y" << std::endl;
  lu_decomposition(Y, L_y, U_y, P); 
//  matrix_print(L_y);
  delete Y;

  Matrix *L_y_truncated = new Matrix(L_y->GetNumRows(), k);
//  Matrix *U_y_truncated = new Matrix(k, U_y->GetNumCols());
  matrix_sliceblock(L_y, L_y_truncated, UPPERLEFT);
//  std::cout << "L_y_truncated" << std::endl;
//  matrix_print(L_y_truncated);
//  matrix_sliceblock(U_y, U_y_truncated, TOPLEFT);
  delete U_y;
  delete L_y;

  Matrix *L_y_t = new Matrix(L_y_truncated->GetNumCols(), L_y_truncated->GetNumRows());
  matrix_transpose(L_y_truncated, L_y_t);
  Matrix *L_y_innerproduct = new Matrix(L_y_truncated->GetNumCols(), L_y_truncated->GetNumCols());
  matrix_multiply(L_y_t, L_y_truncated, L_y_innerproduct);
  GJE_inverse(L_y_innerproduct);
  Matrix *L_y_psuedoinverse = new Matrix(L_y_truncated->GetNumCols(), L_y_truncated->GetNumRows());
  matrix_multiply(L_y_innerproduct, L_y_t, L_y_psuedoinverse);
//  std::cout << "L_y_psuedoinverse: " << std::endl;
//  matrix_print(L_y_psuedoinverse);
  delete L_y_t;
  delete L_y_innerproduct;

  Matrix *B = new Matrix(L_y_psuedoinverse->GetNumRows(), A->GetNumCols());
  Matrix *inter = new Matrix(L_y_psuedoinverse->GetNumRows(), P->GetNumCols());
  matrix_multiply(L_y_psuedoinverse, P, inter);
  matrix_multiply(inter, A, B);
  delete L_y_psuedoinverse;
  delete inter;

  Matrix *L_b = new Matrix(B->GetNumRows(), B->GetNumRows());
  Matrix *U_b = new Matrix(B->GetNumRows(), B->GetNumCols());
//  std::cout << "L_b: " << std::endl;
  lu_columndecomposition(B, L_b, U_b, Q);
//  matrix_print(L_b);
 
  Matrix *L_b_extended = new Matrix(L_b->GetNumRows(), A->GetNumCols());
  L_b_extended->ToIdentity();
  matrix_writeblock(L_b_extended, L_b, UPPERLEFT);
  matrix_multiply(L_y_truncated, L_b_extended, L);
  int num_blocks = (L->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kfix_diagonal<<<num_blocks, THREADS_PER_BLOCK>>>(L);
  cudaDeviceSynchronize();
  
  U->ToZeroes();
  matrix_writeblock(U, U_b, UPPERLEFT);

  delete L_y_truncated;
  delete L_b;
  delete U_b;
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

    matrix_sliceblock(L_loop, L_tl, UPPERLEFT);
    if (L_bl->GetNumRows() > 0)
    {
      matrix_sliceblock(L_loop, L_bl, BOTTOMLEFT);
      matrix_writeblock(L, L_bl, r + i, i);
    }
    //matrix_sliceblock(A_loop, U_tr, UPPERRIGHT);
    matrix_sliceblock(A_loop, U_tl, UPPERLEFT);
//    std::cout << "\nA_copy" << std::endl;
//    matrix_print(A_copy);
//    std::cout << "\nA_tr" << std::endl;
//    matrix_print(A_tr);
    //matrix_writeblock(A_copy, A_loop, BOTTOMRIGHT);
//		matrix_writeblock(P, P_loop, BOTTOMRIGHT);

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
		
//    matrix_sliceblock(P, P_loop, BOTTOMRIGHT);
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
      matrix_sliceblock(A_copy, A_loop, BOTTOMRIGHT);
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
      matrix_writeblock(A_copy, A_loop, BOTTOMRIGHT);
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

__global__ void kfix_diagonal(Matrix *L)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < L->GetNumCols() ? false : true;

  if (!past_length)
  {
    if ((*L)[idx][idx] == 0)
    {
      (*L)[idx][idx] = 1;
    }
  }
}
