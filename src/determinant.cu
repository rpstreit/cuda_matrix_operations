
#include <iostream>
#include <math.h>
#include "common.h"

#include <stack>
#include <tuple>

// sequential implementation of recursive Laplace Expansion
int determinant_recur(Matrix *A)
{
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
		for (int j1 = 0; j1 < level; j1++) // for each column, create a new minor matrix
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
			result += pow(-1, 1 + j1 + 1) * (*A)[0][j1] * determinant(minor_intermediate);

			// free all minor matrices here after recursive call is returning from base case
			delete minor_intermediate;
		}

		// // need to wait for all minor matrices
		// __syncthreads();
	}

	return result;
}

int determinant_iter(Matrix *A)
{
	Matrix * currentMatrix = A;
	int originalLevel = A->GetNumCols();
	int level = originalLevel; // represents current level
	int result;

	// base case
	if (level == 1)
	{
		return (*A)[0][0];
	}
	else if (level == 2)
	{
		return (*A)[0][0] * (*A)[1][1] - (*A)[1][0] * (*A)[0][1];
	}

	int parentIndex = 0; // after level is less than 2

	int[] parentIndices = new int[level - 1]; // 0: [2x2] 1: [3x3] 2: [4x4]
	// parentIndices.reserve(level - 1); // if starting with 4x4, should be 3 levels
	// for (int i = 0; i < level - 1; i++)
	// {
	// 	parentIndices.push_back(0);
	// }

	// pushing <minor matrix, original matrix, col removed, det value>
	std::stack<std::tuple<Matrix *, Matrix *, int, int>> stack;
	for (;;)
	{
		// add all minor matrices to stack
		while (level >= 2)
		{
			for (int j1 = parentIndices[level - 2]; j1 < level; j1++)
			{
				Matrix *minorMatrix = new Matrix(level - 1, level - 1);

				// populate minor matrix
				for (int i = 1; i < level; i++) // always skip first row
				{
					int j2 = 0; 

					// for each col, create a new minor matrix with that col removed
					for (int j = 0; j < level; j++) {
						if (j == j1)
							continue;

						(*minorMatrix)[i - 1][j2] = (*currentMatrix)[i][j];
						j2++;
					}
				}

				stack.push(std::make_tuple(minorMatrix, currentMatrix, j1, 0));
			}

			parentIndices[level - 2]++;
			level--;
			currentMatrix = minorMatrix; // transfer from 4x4 to 3x3
		}

		// get children, calculate determinant for parent
		// there should be 3 2x2 minors, 4 3x3s, etc.
		for (int i = 0; i < topLevel + 1; i++)
		{
			std::tuple<Matrix *, Matrix *, int, int> top = stack.top();
			Matrix * minorMatrix = std::get<0>(top);
			Matrix * parentMatrix = std::get<1>(top);
			int colRemoved = std::get<2>(top);
			int prevDet = std::get<3>(top);

			topLevel = top->GetNumCols();
			stack.pop();

			// C i,j = (-1)^(i + j) * M i,j
			int cofactor = pow(-1, 1 + colRemoved); 
			if (topLevel == 2)
			{
				cofactor = cofactor * (*minorMatrix)[0][0] * (*minorMatrix)[1][1] - (*minorMatrix)[1][0] * (*minorMatrix)[0][1];
			}
			else
			{
				// if greater than 3x3, take from previously calculated det
				cofactor = cofactor * prevDet;
			}

			// result has determinant of parent matrix
			result += cofactor * (*parentMatrix)[1][colRemoved];
		}

		// get parent, set parent with determinant value
		std::tuple<Matrix *, Matrix *, int, int> top = stack.top();
		std::get<3>(top) = result;

		level += 2; // need to return back to parent, (return from 2x2 to 3x3)
		
		// finished
		if (parentIndices[originalLevel - 1] == originalLevel + 2)
		{
			break;
		}
	}
}