

#include <math.h>
#include <iostream>

#include "matrix.h"
#include "common.h"

__global__ void kvector_square(Matrix *src, Matrix *dest);
__global__ void kmatrix_transpose(Matrix *in, Matrix *out);
__global__ void kmatrix_multiply(Matrix *A, Matrix *B, Matrix *result);
__global__ void kmatrix_multiply_mapsums(Matrix *A, Matrix *B, double *result);
__global__ void kmatrix_multiply_reducesums(double *in, int depth, double *out);
__global__ void kmatrix_multiply_writeresult(double *raw, Matrix *result);
__global__ void kmatrix_slicecolumn(Matrix *A, double *slice, int col_idx);
__global__ void kmatrix_writeblock(Matrix *dest, Matrix *src, int tl_row, int tl_col);
__global__ void kmatrix_sliceblock(Matrix *src, Matrix *dest, int tl_row, int tl_col);
__global__ void kmatrix_copy(Matrix *dest, Matrix *src);
__global__ void kmatrix_getelementarymatrix(Matrix *A, Matrix *result, int col);
__global__ void kmatrix_invertelementarymatrix(Matrix *A, Matrix *result, int col);
__global__ void kmatrix_rowswap(Matrix *A, int row1, int row2);
__global__ void kmatrix_subdiagonal_rowswap(Matrix *A, int row1, int row2);
__global__ void kmatrix_subtract(Matrix *A, Matrix *B, Matrix *C);
__global__ void kmatrix_add(Matrix *A, Matrix *B, Matrix *C);
__global__ void kmultiply_scalar(Matrix *output, Matrix *input, double scale);
__global__ void kfloor(Matrix *output, Matrix *input);
__global__ void kmatrix_subdiagonal_writecolumn(Matrix *dest, Matrix *src, int col);

void matrix_subdiagonal_writecolumn(Matrix *dest, Matrix *src, int col)
{
  int num_blocks = (dest->GetNumRows() - col - 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_subdiagonal_writecolumn<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src, col);
  cudaDeviceSynchronize(); 
}

void matrix_rowswap(Matrix *A, int row1, int row2)
{
  int num_blocks = (A->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_rowswap<<<num_blocks, THREADS_PER_BLOCK>>>(A, row1, row2);
  cudaDeviceSynchronize(); 
}

void matrix_subdiagonal_rowswap(Matrix *A, int row1, int row2)
{
  int min = row1 < row2 ? row1 : row2;
  int num_blocks = (min + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_subdiagonal_rowswap<<<num_blocks, THREADS_PER_BLOCK>>>(A, row1, row2);
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
  int num_blocks = (A->GetNumRows() - col - 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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

void matrix_sliceblock(Matrix *src, Matrix *dest, int tl_row, int tl_col)
{
  int cols = dest->GetNumCols();
  int rows = dest->GetNumRows();

  int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  kmatrix_sliceblock<<<num_blocks, THREADS_PER_BLOCK>>>(src, dest, tl_row, tl_col);
  cudaDeviceSynchronize();
}

void matrix_sliceblock(Matrix *src, Matrix *dest, BlockLoc loc)
{
  int tl_row, tl_col;
  switch (loc)
  {
		case UPPERLEFT:
			tl_row = 0;
			tl_col = 0;
			break;

		case UPPERRIGHT:
			tl_row = 0;
			tl_col = src->GetNumCols() - dest->GetNumCols();
			break;

		case BOTTOMLEFT:
			tl_row = src->GetNumRows() - dest->GetNumRows();
			tl_col = 0;
			break;

		case BOTTOMRIGHT:
			tl_row = src->GetNumRows() - dest->GetNumRows();
			tl_col = src->GetNumCols() - dest->GetNumCols();
			break;
  }
  matrix_sliceblock(src, dest, tl_row, tl_col);
}

void matrix_writeblock(Matrix *dest, Matrix *src, int tl_row, int tl_col)
{
  int cols = src->GetNumCols();
  int rows = src->GetNumRows();

  int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_writeblock<<<num_blocks, THREADS_PER_BLOCK>>>(dest, src, tl_row, tl_col);
  cudaDeviceSynchronize();
}

void matrix_writeblock(Matrix *dest, Matrix *src, BlockLoc loc)
{
	int tl_row, tl_col;
  switch (loc)
  {
		case UPPERLEFT:
			tl_row = 0;
			tl_col = 0;
			break;

		case UPPERRIGHT:
			tl_row = 0;
			tl_col = dest->GetNumCols() - src->GetNumCols();
			break;

		case BOTTOMLEFT:
			tl_row = dest->GetNumRows() - src->GetNumRows();
			tl_col = 0;
			break;

		case BOTTOMRIGHT:
			tl_row = dest->GetNumRows() - src->GetNumRows();
			tl_col = dest->GetNumCols() - src->GetNumCols();
			break;
  }
	matrix_writeblock(dest, src, tl_row, tl_col);
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
  if (A->GetNumCols() != B->GetNumRows()
      || A->GetNumRows() != result->GetNumRows()
      || B->GetNumCols() != result->GetNumCols())
  {
    std::cerr << "error: matrix_multiply input/output dimensions are inconsistent" << std::endl;
    exit(EXIT_FAILURE);
  }
  double *inter1, *inter2;

  int rrows = A->GetNumRows();
  int rdepth = A->GetNumCols();
  int rcols = B->GetNumCols();
	
  int num_blocks = (rrows * rdepth * rcols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaMalloc((void **) &inter1, rrows * rdepth * rcols * sizeof(double));
  kmatrix_multiply_mapsums<<<num_blocks, THREADS_PER_BLOCK>>>(A, B, inter1);
cudaDeviceSynchronize();
//  double h_inter1[rrows * rdepth * rcols];
//  cudaMemcpy(h_inter1, inter1, rrows * rdepth * rcols *sizeof(double), cudaMemcpyDeviceToHost);
//  for (int i = 0; i < rrows; ++i)
//  {
//    std::cout<<"[";
//    for (int j = 0; j < rcols; ++j)
//    {
//      std::cout << " {";
//      for (int k = 0; k < rdepth; ++k)
//      {
//        std::cout << " " << h_inter1[i * rcols * rdepth + j * rdepth + k];
//      }
//      std::cout << " }";
//    }
//    std::cout << " ]\n";
//  }

  num_blocks = (rdepth + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * rrows * rcols;
  cudaMalloc((void **) &inter2, rrows * rdepth * rcols * sizeof(double));
  for(;;)
  {
    //std::cout << "num_blocks: " << num_blocks << std::endl;
    kmatrix_multiply_reducesums<<<num_blocks, THREADS_PER_BLOCK>>>(inter1, rdepth, inter2);
    cudaDeviceSynchronize();
//    cudaMemcpy(h_inter1, inter2, rrows *(num_blocks/(rrows*rcols))* rcols *sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < rrows; ++i)
//    {
//      std::cout<<"[";
//      for (int j = 0; j < rcols; ++j)
//      {
//        std::cout << " {";
//        for (int k = 0; k < (num_blocks/(rrows*rcols)) ; ++k)
//        {
//          std::cout << " " << h_inter1[i * rcols * (num_blocks/(rrows*rcols)) + j * (num_blocks/(rrows*rcols)) + k];
//        }
//        std::cout << " }";
//      }
//      std::cout << " ]\n";
//    }
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
  cudaDeviceSynchronize();

  cudaFree(inter2);
  cudaFree(inter1);
}

void matrix_print(Matrix *A)
{
  std::cout << "Matrix@" << A << ":\tnum_rows: " << A->GetNumRows() << "\tnum_cols: " << A->GetNumCols() << std::endl;
  int i, j;
  for (i = 0; i < A->GetNumRows(); ++i)
  {
    std::cout << "[";
    for (j = 0; j < A->GetNumCols(); ++j)
    {
      std::cout << " " << (*A)[i][j];
    }
    std::cout << " ]" << std::endl;
  }
}

bool matrix_equals(Matrix *A, Matrix *B, double error)
{
  int i, j;

  cudaDeviceSynchronize();
  if (A->GetNumRows() != B->GetNumRows()
   || A->GetNumCols() != B->GetNumCols())
  {
    return false;
  }

  for (i = 0; i < A->GetNumRows(); ++i)
  {
    for (j = 0; j < A->GetNumCols(); ++j)
    {
      int diff = (*A)[i][j] - (*B)[i][j];
      diff = diff < 0 ? diff * -1.f : diff;
      if (diff > error)
      {
        return false;
      }
    }
  }

  return true;
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

void matrix_subtract(Matrix *A, Matrix *B, Matrix *C)
{
	int num_blocks = (C->GetNumRows() * C->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_subtract<<<num_blocks, THREADS_PER_BLOCK>>>(A, B, C);
  cudaDeviceSynchronize();
}

void matrix_add(Matrix *A, Matrix *B, Matrix *C)
{
	int num_blocks = (C->GetNumRows() * C->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmatrix_add<<<num_blocks, THREADS_PER_BLOCK>>>(A, B, C);
  cudaDeviceSynchronize();
}

void matrix_multiply_scalar(Matrix *output, Matrix *input, double scale) {
  int num_blocks = (input->GetNumRows() * input->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  // std::cout << "blocks are:" << num_blocks << std::endl;
  kmultiply_scalar<<<num_blocks, THREADS_PER_BLOCK>>>(output, input, scale);
  cudaDeviceSynchronize();
}

void matrix_floor_small(Matrix* output, Matrix *input) {
  int num_blocks = (input->GetNumRows() * input->GetNumCols() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  // std::cout << "blocks are:" << num_blocks << std::endl;
  kfloor<<<num_blocks, THREADS_PER_BLOCK>>>(output, input);
  cudaDeviceSynchronize();
}

__global__ void kfloor(Matrix *output, Matrix *input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double * outputM = output->GetFlattened();
  double * inputM = input->GetFlattened();
  if(idx < (input->GetNumCols() * input->GetNumRows())) 
  {
    if(inputM[idx] < 0.00001)
      outputM[idx] = 0;
    else
      outputM[idx] = inputM[idx];
  }
}

__global__ void kmultiply_scalar(Matrix *output, Matrix *input, double scale)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double * outputM = output->GetFlattened();
  double * inputM = input->GetFlattened();
  if(idx < (input->GetNumCols() * input->GetNumRows())) 
  {
    outputM[idx] = inputM[idx] * scale;
  }
}

__global__ void kmatrix_subdiagonal_writecolumn(Matrix *dest, Matrix *src, int col)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < dest->GetNumRows() - col - 1? false : true;

  if (!past_length)
  {
    (*dest)[col + idx + 1][col] = (*src)[col + idx + 1][col];
  }
}

__global__ void kmatrix_rowswap(Matrix *A, int row1, int row2)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < A->GetNumCols() ? false : true;

  if (!past_length)
  {
    double temp = (*A)[row1][idx];
    (*A)[row1][idx] = (*A)[row2][idx];
    (*A)[row2][idx] = temp;
  }
}

__global__ void kmatrix_subtract(Matrix *A, Matrix *B, Matrix *C)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < C->GetNumRows() * C->GetNumCols() ? false : true;

  if (!past_length)
  {
    C->GetFlattened()[idx] = A->GetFlattened()[idx] - B->GetFlattened()[idx];
  }
}

__global__ void kmatrix_add(Matrix *A, Matrix *B, Matrix *C)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < C->GetNumRows() * C->GetNumCols() ? false : true;

  if (!past_length)
  {
    C->GetFlattened()[idx] = A->GetFlattened()[idx] + B->GetFlattened()[idx];
  }
}

__global__ void kmatrix_subdiagonal_rowswap(Matrix *A, int row1, int row2)
{
  int min = row1 < row2 ? row1 : row2;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < min ? false : true;

  if (!past_length)
  {
    double temp = (*A)[row1][idx];
    (*A)[row1][idx] = (*A)[row2][idx];
    (*A)[row2][idx] = temp;
  }
}

__global__ void kmatrix_invertelementarymatrix(Matrix *A, Matrix *result, int col)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < result->GetNumRows() - col - 1? false : true;

  if (!past_length)
  {
    (*result)[col + idx + 1][col] = (*A)[col + idx + 1][col] * -1.f;
  }
}

__global__ void kmatrix_getelementarymatrix(Matrix *A, Matrix *result, int col)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool past_length = idx < result->GetNumRows() - col ? false : true;

  if (!past_length && idx != 0)
  {
    double pivot = (*A)[col][col];
    (*result)[col + idx][col] = -1.f * (*A)[col + idx][col] / pivot;
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
  // std::cout << "Performing dot product" << std::endl;
  // First Transpose vector 2 for matrix multiplication
  int length = vec1->GetNumRows();
  Matrix *temp = new Matrix(1, length);
  matrix_transpose(vec2, temp);
  // std::cout << "Temp:" << std::endl;
  // matrix_print(temp);

  // Perform the multiplication
  Matrix *result = new Matrix(1, 1);
  matrix_multiply(temp, vec1, result);
  // std::cout << "Result:" << std::endl;
  // matrix_print(result);

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
  // std::cout << "Inside norm function" << std::endl;
  // matrix_print(vector);
  kvector_square<<<(length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vector, output_vector);
  cudaDeviceSynchronize();
  // matrix_print(output_vector);
  // matrix_print(output_vector);
  double addedSum = reduce(output_vector->GetFlattened(), length, ADD);
  // std::cout << "Inside Added sum " << addedSum << std::endl;

  double normal =  sqrt(addedSum);
  delete output_vector;
  return normal;
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


__global__ void kmatrix_sliceblock(Matrix *src, Matrix *dest, int tl_row, int tl_col)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;
  
  int row = idx / dest->GetNumCols();
  int col = idx % dest->GetNumCols();

  if (!past_length)
  {
    (*dest)[row][col] = (*src)[tl_row + row][tl_col + col];
  }
}

__global__ void kmatrix_writeblock(Matrix *dest, Matrix *src, int tl_row, int tl_col)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < src->GetNumRows() * src->GetNumCols() ? false : true;

  int row = idx / src->GetNumCols();
  int col = idx % src->GetNumCols();

  if (!past_length)
  {
    (*dest)[tl_row + row][tl_col + col] = (*src)[row][col];
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
    slice[idx] = (*A)[idx][col_idx];
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
  int blocks_per_depth = (depth + blockDim.x - 1) / blockDim.x;
  int cubby = (blockIdx.x / blocks_per_depth) * depth;
  int block_in_depth = blockIdx.x % blocks_per_depth;
  int cubby_idx = block_in_depth * blockDim.x + threadIdx.x;
  int idx = 0;

  __shared__ double s_data[THREADS_PER_BLOCK];

  if (cubby_idx < depth)
  {
    idx = cubby + cubby_idx;
    s_data[threadIdx.x] = in[idx];
  }
  else
  {
    s_data[threadIdx.x] = 0.f;
  }
  __syncthreads();
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
  {
    if (tid < i && (cubby_idx + i) < depth)
    {
      s_data[tid] = s_data[tid] + s_data[tid + i];
    }

    __syncthreads(); // wait for round to finish
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
    int depth = idx % (A->GetNumCols());

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
