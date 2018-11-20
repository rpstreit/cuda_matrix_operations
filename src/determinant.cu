
#include <iostream>
#include "matrix.h"
#include <math.h>

#define THREADS_PER_BLOCK 256

// void matrix_createminor(Matrix *dest, Matrix *src, int row, int col)
// {
// 	int cols = dest->GetNumCols();

// 	int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// 	kmatrix_minor<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src, cols - 1);
// }

// __global__ void kmatrix_determinant(Matrix *A)
// {
// 	// A is a NxN matrix

// 	int levels = A->GetNumCols();
// 	__shared__ int det; // sharing N - 1 determinants results 

// 	// outer for loop to parallelize
// 	for (int i = 0; i < levels; i++)
// 	{
// 		// 

// 		// call kmatrix_determinant_recur here N times
// 		// should be N distinct recursive calls
// 	}

// 	__syncthreads(); // wait for all N threads to finish

// 	free(shared_det);
// 	// at end, copy shared_det to return and then free
// }


// __global__ void kmatrix_minor(Matrix* dest, Matrix *src, int col)
// {
// 	int level = dest->getNumCols();

// 	for (int i = 1; i < level - 1; i++)
// 	{
// 		for (int j = 0; j < level; j++)
// 		{
// 			if (j == )
// 		}

// 	}
// }

// needs __host__ so host can call it
// needs __device__ since CUDA supports recursive calls only for __device__ functions
__device__ int kmatrix_determinant_recur(Matrix *A, Matrix )
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // ?
	int stride = blockDim.x * gridDim.x;

	int level = A->GetNumCols(); // represents current level
	int result;

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
		for (int j1 = idx; j1 < level; j1 += stride) // for each column, create a new minor matrix
		{
			// create minor matrix of size N - 1 here, allocate memory here
			// pass it into kmatrix_minor and have it populate it
			Matrix *minor_intermediate = new Matrix(level - 1, level - 1);

			for (int i = 1; i < level; i++) // always skip first row
			{
				int j2 = 0; 

				// for each col, create a new minor matrix
				for (int j = 0; j < level; j++) {
					if (j == j1)
						continue;

					(*minor_intermediate)[i - 1][j2] = (*A)[i][j];
					j2++;
				}
			}

			// calculate determinant from (N - 1)-size minor matrix
			result += pow(-1, 1 + j1 + 1) * (*A)[0][j1] * kmatrix_determinant_recur(minor_intermediate);

			// free all minor matrices here after recursive call is returning from base case
			delete minor_intermediate;
		}

		// need to wait for all minor matrices
		__syncthreads();
	}

	return result;
}

__global__ void kmatrix_determinant(Matrix *A)
{
	kmatrix_determinant_recur(A);
}

void determinant(Matrix *A)
{
	int rows = A->GetNumRows();
	int cols = A->GetNumCols();
	int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kmatrix_determinant<<<num_blocks, THREADS_PER_BLOCK>>>(A);
}