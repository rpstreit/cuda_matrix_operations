
#include <tuple>

#include "matrix.h"
#include "common.h"
#include <math.h>

__global__ void kmatrix_transpose(Matrix *in, Matrix *out);
__global__ void kmatrix_multiply(Matrix *A, Matrix *B, Matrix *result);
__global__ void kmatrix_multiply_mapsums(Matrix *A, Matrix *B, double *result);
__global__ void kmatrix_multiply_reducesums(double *in, int depth, double *out);
__global__ void kmatrix_multiply_writeresult(double *raw, Matrix *result);
__global__ void kmatrix_slicecolumn(Matrix *A, double *slice, int col_idx);
__global__ void kmatrix_writeblock(Matrix *dest, Matrix *src, BlockLoc loc);
__global__ void kmatrix_sliceblock(Matrix *src, Matrix *dest, BlockLoc loc);
__global__ void kmatrix_copy(Matrix *dest, Matrix *src);
__global__ void kmatrix_getelementarymatrix(Matrix *A, Matrix *result, int col);
__global__ void kmatrix_invertelementarymatrix(Matrix *A, Matrix *result, int col);
__global__ void kmatrix_rowswap(Matrix *A, int row1, int row2);

void matrix_rowswap(Matrix *A, int row1, int row2)
{
  int num_blocks = (A->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_rowswap<<<num_blocks, THREADS_PER_BLOCK>>>(A, row1, row2);
  cudaDeviceSynchronize(); 
}

void matrix_getelementarymatrix(Matrix *A, Matrix *result, int col)
{
  result->ToIdentity();

  int num_blocks = (A->GetNumRows() - col + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_getelementarymatrix<<<num_blocks, THREADS_PER_BLOCK>>>(A, result, col);
  cudaDeviceSynchronize();
}

void matrix_invertelementarymatrix(Matrix *A, Matrix *result, int col)
{
  result->ToIdentity();
  int num_blocks = (A->GetNumRows() - col + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_invertelementarymatrix<<<num_blocks, THREADS_PER_BLOCK>>>(A, result, col);
  cudaDeviceSynchronize();
}

void matrix_copy(Matrix *dest, Matrix *src)
{
  int cols = dest->GetNumCols();
  int rows = dest->GetNumRows();

  int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
 
  kmatrix_copy<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src);
  cudaDeviceSynchronize();
}

__global__ void kvector_square(Matrix *src, Matrix *dest);

void matrix_sliceblock(Matrix *src, Matrix *dest, BlockLoc loc)
{
  int cols = dest->GetNumCols();
  int rows = dest->GetNumRows();

  int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  kmatrix_sliceblock<<<num_blocks, THREADS_PER_BLOCK>>>(src, dest, loc);
  cudaDeviceSynchronize();
}

void matrix_writeblock(Matrix *dest, Matrix *src, BlockLoc loc)
{
  int cols = src->GetNumCols();
  int rows = src->GetNumRows();

  int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_writeblock<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src, loc);
  cudaDeviceSynchronize();
}

void matrix_slicecolumn(Matrix *A, double *slice, int col_idx)
{
	int num_blocks = (A->GetNumRows() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_slicecolumn<<<num_blocks, THREADS_PER_BLOCK>>>(A, slice, col_idx);
  cudaDeviceSynchronize();
}

// matrix_multiply
//
// Computes a matrix matrix multiplication in parallel
//
// Inputs: Managed matrices A and B pointers, pre allocated matrix
// result pointer. Note: ensure that for mxl * lxn matrix multiply
// result is mxn
// Outpus: Resulting multiply in result
void matrix_multiply(Matrix *A, Matrix *B, Matrix *result)
{ 
  double *inter1, *inter2;

  int rrows = A->GetNumRows();
  int rdepth = A->GetNumCols();
  int rcols = B->GetNumCols();
	
  int num_blocks = (rrows * rdepth * rcols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaMalloc((void **) &inter1, rrows * rdepth * rcols * sizeof(double));
  kmatrix_multiply_mapsums<<<num_blocks, THREADS_PER_BLOCK>>>(A, B, inter1);
  cudaDeviceSynchronize();

  cudaMalloc((void **) &inter2, num_blocks * sizeof(double));
  num_blocks = (rdepth + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * rrows * rcols;
  for(;;)
  {
    kmatrix_multiply_reducesums<<<num_blocks, THREADS_PER_BLOCK>>>(inter1, rdepth, inter2);
    cudaDeviceSynchronize();
    if (num_blocks == rrows * rcols)
    {
      break;
    }
    else
    {
      double *temp = inter1;
      inter1 = inter2;
      inter2 = temp;
      rdepth = num_blocks / (rrows * rcols);
      num_blocks = (rdepth + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * rrows * rcols;
    }
  }

  num_blocks = (rrows * rcols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_multiply_writeresult<<<num_blocks, THREADS_PER_BLOCK>>>(inter2, result);

  cudaFree(inter2);
  cudaFree(inter1);
}

// matrix_transpose
//
// Computes matrix transpose in parallel
//
// Inputs: Managed matrix pointer, pre allocated managed matrix 
// result pointer. Note: ensure that for mxn matrix mat result is 
// nxm
// Outputs: Resulting transpose in result
void matrix_transpose(Matrix *mat, Matrix *result)
{
	int num_blocks = (mat->GetNumRows() * mat->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_transpose<<<num_blocks, THREADS_PER_BLOCK>>>(mat, result);
  cudaDeviceSynchronize();
}

__global__ void kmatrix_rowswap(Matrix *A, int row1, int row2)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < A->GetNumCols() ? false : true;

  if (!past_length && idx != 0)
  {
    double temp = (*A)[row1][idx];
    (*A)[row1][idx] = (*A)[row2][idx];
    (*A)[row2][idx] = temp;
  }
}

__global__ void kmatrix_invertelementarymatrix(Matrix *A, Matrix *result, int col)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < result->GetNumRows() - col ? false : true;

  if (!past_length && idx != 0)
  {
    (*result)[col + idx][col] = (*A)[col + idx][col] * -1.f;
  }
}

__global__ void kmatrix_getelementarymatrix(Matrix *A, Matrix *result, int col)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < result->GetNumRows() - col ? false : true;

  if (!past_length && idx != 0)
  {
    double pivot = (*A)[col][col];
    (*result)[col + idx][col] = (*A)[col + idx][col] / pivot;
  }
}

/**
 * Performs the dot product of two vectors
 * Assumes both vectors are column vectors (columns = 1) 
 *   and equal length
 * Makes use of transpose and matrix multiply
 * @param  vec1 Vector of size rows x 1
 * @param  vec2 Vector of size rows x 1
 * @return      dot product of the two vectors.....
 */
double dot_product(Matrix *vec1, Matrix *vec2) 
{
  // First Transpose vector 2 for matrix multiplication
  int length = vec1->GetNumRows();
  Matrix *temp = new Matrix(1, length);
  matrix_transpose(vec2, temp);
  
  // Perform the multiplication
  Matrix *result = new Matrix(1, 1);
  matrix_multiply(vec1, temp, result);

  // Resulting 1x1 matrix holds the dot product
  double prod = *(result->GetFlattened());
  delete temp;
  delete result;
  return prod;
}


/**
 * Get the norm of a vector (i.e. the magnitude-ish)
 * This will be the |v|
 * @param  vector vector to get norm of
 * @return        the weird
 */
double norm(Matrix *vector) 
{
  // Make a new vector
  Matrix *output_vector = new Matrix(vector->GetNumRows(), vector->GetNumCols());
  int length;
  if(vector->GetNumRows() == 1) {
    length = vector->GetNumCols();
  } else {
    length = vector->GetNumRows();
  }

  kvector_square<<<length / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vector, output_vector);

  double norm =  sqrt(reduce(output_vector, length, Reduction::ADD));
  delete output_vector;
  return norm;
}


/**
 * Squares every element in a vector
 * Assumes destination vector has been properly allocated
 * @param src  source vector of elements to square
 * @param dest output vector of squared elements
 */
__global__ void kvector_square(Matrix *src, Matrix *dest)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int length;
  if(src->GetNumRows() == 1) {
    length = src->GetNumCols();
  } else {
    length = src->GetNumRows();
  }

  if(idx < length) {
    dest->GetFlattened()[idx] = src->GetFlattened()[idx] * src->GetFlattened()[idx];
  }

}


__global__ void kmatrix_sliceblock(Matrix *src, Matrix *dest, BlockLoc loc)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;
  
  int row = idx / dest->GetNumCols();
  int col = idx % dest->GetNumCols();

  int start_row, start_col;
  if (!past_length)
  {
    switch(loc)
    {
      case BlockLoc::UPPERLEFT:
        start_row = 0;
        start_col = 0;
        break;

      case BlockLoc::UPPERRIGHT:
        start_row = 0;
        start_col = src->GetNumCols() - dest->GetNumCols();
        break;

      case BlockLoc::BOTTOMLEFT:
        start_row = src->GetNumRows() - dest->GetNumRows();
        start_col = 0;
        break;

      case BlockLoc::BOTTOMRIGHT:
        start_row = src->GetNumRows() - dest->GetNumRows();
        start_col = src->GetNumCols() - dest->GetNumCols();
        break;
    }
    (*dest)[row][col] = (*src)[start_row + row][start_col + col];
  }
}

__global__ void kmatrix_writeblock(Matrix *dest, Matrix *src, BlockLoc loc)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < src->GetNumRows() * src->GetNumCols() ? false : true;

  int row = idx / src->GetNumCols();
  int col = idx % src->GetNumCols();

  int start_row, start_col;
  if (!past_length)
  {
    switch(loc)
    {
      case BlockLoc::UPPERLEFT:
        start_row = 0;
        start_col = 0;
        break;

      case BlockLoc::UPPERRIGHT:
        start_row = 0;
        start_col = dest->GetNumCols() - src->GetNumCols();
        break;

      case BlockLoc::BOTTOMLEFT:
        start_row = dest->GetNumRows() - src->GetNumRows();
        start_col = 0;
        break;

      case BlockLoc::BOTTOMRIGHT:
        start_row = dest->GetNumRows() - src->GetNumRows();
        start_col = dest->GetNumCols() - src->GetNumCols();
        break;
    }
    (*dest)[start_row + row][start_col + col] = (*src)[row][col];
  }
}


__global__ void kmatrix_transpose(Matrix *in, Matrix *out)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < in->GetNumRows() * in->GetNumCols() ? false : true;

  if (!past_length)
  {
    int row = idx / in->GetNumCols();
    int col = idx % in->GetNumCols();

    (*out)[col][row] = (*in)[row][col];
  }
}

__global__ void kmatrix_slicecolumn(Matrix *A, double *slice, int col_idx)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < A->GetNumRows())
  {
    slice[idx] = (*A)[col_idx][idx];
  }
}

__global__ void kmatrix_multiply_writeresult(double *raw, Matrix *result)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < result->GetNumRows() * result->GetNumCols() ? false : true;

  if (!past_length)
  {
    result->GetFlattened()[idx] = raw[idx];
  }
}

__global__ void kmatrix_multiply_reducesums(double *in, int depth, double *out)
{
  int tid = threadIdx.x;
  int blocksPerDepth = (depth + blockDim.x - 1) / blockDim.x;
  int idx_2d = blockIdx.x / blocksPerDepth;
  int base = idx_2d * depth;
  int idx, block_idx, depth_idx;
  int block_in_depth;
 
  __shared__ double s_data[THREADS_PER_BLOCK];

  if (!(blockDim.x * blocksPerDepth + threadIdx.x > depth))
  {
    block_in_depth = blockIdx.x % blocksPerDepth;
    block_idx = base + block_in_depth * blocksPerDepth;
    idx = block_idx + threadIdx.x;
    depth_idx = block_in_depth * blockDim.x + threadIdx.x;

    s_data[threadIdx.x] = in[idx];
  }
  __syncthreads();
  if (!(blockDim.x * blocksPerDepth + threadIdx.x > depth))
  {
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (threadIdx.x < i && (depth_idx + i) < depth)
			{
				s_data[tid] = s_data[tid] + s_data[tid + i];
      }

			__syncthreads(); // wait for round to finish
		}	
  }
	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}

__global__ void kmatrix_multiply_mapsums(Matrix *A, Matrix *B, double *result)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < A->GetNumRows() * B->GetNumCols() * A->GetNumCols() ? false : true;
  
  if (!past_length)
  {
    int row = idx / (B->GetNumCols() * A->GetNumCols());
    int col = (idx / A->GetNumCols()) % B->GetNumCols();
    int depth = idx % (B->GetNumCols() * A->GetNumCols());

    result[idx] = (*A)[row][depth] * (*B)[depth][col];
  }
}

__global__ void kmatrix_copy(Matrix *dest, Matrix *src)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;

  if (!past_length)
  {
    dest->GetFlattened()[idx] = src->GetFlattened()[idx];
  }
}
