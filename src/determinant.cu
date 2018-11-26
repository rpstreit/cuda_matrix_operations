
#include <math.h>
#include <iostream>
#include "common.h"
#include "matrix_inverse.h"
#include "lu_decomposition.h"

#include <stack>
//#include <tuple>

// sequential implementation of recursive Laplace Expansion
double determinant_recur(Matrix *A)
{
	int level = A->GetNumCols(); // represents current level
	double result;

	if (level == 1)
	{
		return (*A)[0][0];
	}
	else if (level == 2)
	{
		return (*A)[0][0] * (*A)[1][1] - (*A)[1][0] * (*A)[0][1];
	}
	else 
	{
		result = 0;

		// parallelize (N - 1)-size minor matrix computations
		for (int j1 = 0; j1 < level; j1++) // for each column, create a new minor matrix
		{
			// create minor matrix of size N - 1 here, allocate memory here
			// pass it into kmatrix_minor and have it populate it
			Matrix *minor_inter = new Matrix(level - 1, level - 1);

			for (int i = 1; i < level; i++) // always skip first row
			{
				int j2 = 0; 

				// for each col, create a new minor matrix
				for (int j = 0; j < level; j++) {
					if (j == j1)
						continue;

					(*minor_inter)[i - 1][j2] = (*A)[i][j];
					j2++;
				}
			}

			// calculate determinant from (N - 1)-size minor matrix
			double inter_value = determinant_recur(minor_inter);
			//printf("inter_value %f\n", inter_value);	
			result += pow(-1, 1 + j1 + 1) * (*A)[0][j1] * inter_value;

			// free all minor matrices here after recursive call is returning from base case
			delete minor_inter;
		}

		// // need to wait for all minor matrices
		// __syncthreads();
	}

	return ceil(result);
}

__global__ void kmatrix_getrowindex(Matrix *input, int *index, int row)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	double * inputM = input->GetFlattened();
	if (idx < input->GetNumCols()) // only getting first row
	{
		if (inputM[idx + row*input->GetNumCols()] == 1)
		{
			*index = idx;
		}

		//if ((*input)[row][idx] == 1)
		//{
		//	*index = idx;
		//}		
	}	
}

double matrix_getpermutationdeterminant(Matrix *input)
{
	int num_rowswaps = 0;	
	int *index;
	cudaMalloc((void **) &index, sizeof(int));

	matrix_print(input);
	double * inputM = input->GetFlattened();
	
	for (int i = 0; i < input->GetNumCols() * input->GetNumRows(); i++)
	{
		printf("%f, ", inputM[i]);
	}
	printf("\n");

	int num_blocks = (input->GetNumCols() + THREADS_PER_BLOCK + 1) / THREADS_PER_BLOCK;

	int row_index = 0;
	while (row_index < input->GetNumRows())	
	{
		// find index of 1 for this row
		kmatrix_getrowindex<<<num_blocks, THREADS_PER_BLOCK>>>(input, index, row_index);
		cudaDeviceSynchronize();		

		int index_out;
		cudaMemcpy(&index_out, index, sizeof(int), cudaMemcpyDeviceToHost);
		printf("index: %d row %d\n", index_out, row_index);

		// if current row is not correct, swap into correct location	
		if (index_out != row_index)
		{
			matrix_rowswap(input, index_out, row_index); 	
			num_rowswaps += 1;
		}
		else
		{
			row_index += 1; // if correct, no need to swap this row again
		}
	}	

	cudaFree(index);

	return num_rowswaps;
}

double matrix_diagonalproduct(Matrix *input)
{
	double product = 1;
	double * inputM = input->GetFlattened();

	for (int i = 0; i < input->GetNumCols(); i++)
	{
		product *= inputM[i + i*input->GetNumCols()];
	}

	return product;
}

double determinant_lu(Matrix *A)
{
	Matrix *P = new Matrix(A->GetNumCols(), A->GetNumCols());
	Matrix *L = new Matrix(A->GetNumCols(), A->GetNumCols());
	Matrix *U = new Matrix(A->GetNumRows(), A->GetNumCols());

	lu_decomposition(A, L, U, P);

	int P_det = matrix_getpermutationdeterminant(P);
	P_det = pow(-1, P_det);
	double L_det = matrix_diagonalproduct(L);
	double U_det = matrix_diagonalproduct(U);

  	printf("P_det %d, L_det %d, U_det %d\n", P_det, L_det, U_det);

	delete P;
	delete L;
	delete U;

	return P_det * L_det * U_det;		
}

// int determinant_iter(Matrix *A)
// {
// 	Matrix * currentMatrix = A;
// 	int originalLevel = A->GetNumCols();
// 	int level = originalLevel; // represents current level
// 	int result;

// 	stack.push(std::make_tuple(A, A, 0, 0));

// 	// base case
// 	if (level == 1)
// 	{
// 		return (*A)[0][0];
// 	}
// 	else if (level == 2)
// 	{
// 		return (*A)[0][0] * (*A)[1][1] - (*A)[1][0] * (*A)[0][1];
// 	}

// 	int parentIndex = 0; // after level is less than 2

// 	int[] parentIndices = new int[level - 1]; // 0: [2x2] 1: [3x3] 2: [4x4]
// 	// parentIndices.reserve(level - 1); // if starting with 4x4, should be 3 levels
// 	// for (int i = 0; i < level - 1; i++)
// 	// {
// 	// 	parentIndices.push_back(0);
// 	// }

// 	// pushing <minor matrix, original matrix, col removed, det value>
// 	std::stack<std::tuple<Matrix *, Matrix *, int, int>> stack;
// 	for (;;)
// 	{
// 		// add all minor matrices to stack
// 		while (level >= 2)
// 		{
// 			for (int j1 = parentIndices[level - 2]; j1 < level; j1++)
// 			{
// 				Matrix *minorMatrix = new Matrix(level - 1, level - 1);

// 				// populate minor matrix
// 				for (int i = 1; i < level; i++) // always skip first row
// 				{
// 					int j2 = 0; 

// 					// for each col, create a new minor matrix with that col removed
// 					for (int j = 0; j < level; j++) {
// 						if (j == j1)
// 							continue;

// 						(*minorMatrix)[i - 1][j2] = (*currentMatrix)[i][j];
// 						j2++;
// 					}
// 				}

// 				stack.push(std::make_tuple(minorMatrix, currentMatrix, j1, 0));
// 			}

// 			parentIndices[level - 2]++;
// 			level--;
// 			currentMatrix = minorMatrix; // transfer from 4x4 to 3x3
// 		}

// 		// get children, calculate determinant for parent
// 		// there should be 3 2x2 minors, 4 3x3s, etc.
// 		for (int i = 0; i < topLevel + 1; i++)
// 		{
// 			std::tuple<Matrix *, Matrix *, int, int> top = stack.top();
// 			Matrix * minorMatrix = std::get<0>(top);
// 			Matrix * parentMatrix = std::get<1>(top);
// 			int colRemoved = std::get<2>(top);
// 			int prevDet = std::get<3>(top);

// 			topLevel = top->GetNumCols();
// 			stack.pop();

// 			// C i,j = (-1)^(i + j) * M i,j
// 			int cofactor = pow(-1, 1 + colRemoved); 
// 			if (topLevel == 2)
// 			{
// 				cofactor = cofactor * (*minorMatrix)[0][0] * (*minorMatrix)[1][1] - (*minorMatrix)[1][0] * (*minorMatrix)[0][1];
// 			}
// 			else
// 			{
// 				// if greater than 3x3, take from previously calculated det
// 				cofactor = cofactor * prevDet;
// 			}

// 			// result has determinant of parent matrix
// 			result += cofactor * (*parentMatrix)[1][colRemoved];
// 		}

// 		// get parent, set parent with determinant value
// 		std::tuple<Matrix *, Matrix *, int, int> top = stack.top();
// 		std::get<3>(top) = result;

// 		level += 2; // need to return back to parent, (return from 2x2 to 3x3)
		
// 		// finished
// 		if (parentIndices[originalLevel - 1] == originalLevel + 2)
// 		{
// 			std::tuple<Matrix *, Matrix *, int, int> top = stack.top();
// 			return std::get<3>(top);
// 		}
// 	}
// }
