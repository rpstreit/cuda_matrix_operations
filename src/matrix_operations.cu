
#include "matrix.h"
#include "common.h"

__global__ void kmatrix_transpose(Matrix *in, Matrix *out);
__global__ void kmatrix_multiply(Matrix *A, Matrix *B, Matrix *result)
__global__ void kmatrix_multiply_mapsums(Matrix *A, Matrix *B, double *result)
__global__ void kmatrix_multiply_reducesums(double *in, int depth, double *out)
__global__ void kmatrix_multiply_writeresult(double *raw, Matrix *result)

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

  int num_blocks = (rrows * rcols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
  matrix_transpose<<<num_blocks, THREADS_PER_BLOCK>>>(mat, result);
  cudaDeviceSynchronize();
}

__global__ kmatrix_transpose(Matrix *in, Matrix *out)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < in->GetNumRows() * in->GetNumCols() ? false : true;

  if (!past_length)
  {
    int row = idx / in->GetNumCols();
    int col = idx % in->GetNumCols();

    out[col][row] = in[row][col];
  }
}

__global__ void kmatrix_multiply_writeresult(double *raw, Matrix *result)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < result->GetNumRows() * result->GetNumCols() ? false : true;

  result.GetRaw()[idx] = raw[idx];
}

__global__ void kmatrix_multiply_reducesums(double *in, int depth, double *out)
{
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
	int tid = threadIdx.x;
	bool past_length = idx < A->GetNumRows() * B->GetNumCols() * A->GetNumCols() ? false : true;
  
  if (!past_length)
  {
    int row = idx / (B->GetNumCols() * A->GetNumCols());
    int col = (idx / A->GetNumCols()) % B->GetNumCols();
    int depth = idx % (B->GetNumCols() * A->GetNumCols());

    result[idx] = A[row][depth] * B[depth][col];
  }
}
