
#include <fstream>

#include "matrix.h"
#include "common.h"

__global__ void kset_zeroes(double *A);
__global__ void kset_identity(double *A, int cols);

Matrix::Matrix(const char *file) :
  num_rows(0),
  num_cols(0),
  flat(0)
{
  this->Parse(file);
}

Matrix::Matrix(const Matrix &copy) :
  num_rows(copy.num_rows),
  num_cols(copy.num_cols),
  flat(0)
{
  cudaMallocManaged(&flat, sizeof(double) * copy.num_rows * copy.num_cols);
  for (int i = 0; i < this->num_rows; ++i)
  {
    for (int j = 0; j < this->num_cols; ++j)
    {
      this->flat[i * copy.num_cols + j] = copy.flat[i * copy.num_cols + j];
    }
  }
}

Matrix::Matrix(int num_rows, int num_cols, bool identity) :
  num_rows(num_rows),
  num_cols(num_cols),
  flat(0)
{
  cudaMallocManaged(&flat, sizeof(double) * num_rows * num_cols);

  if (identity)
  {
    ToIdentity();
  }
}

Matrix::~Matrix(void)
{ 
  cudaFree(this->flat);
}


/////////////////////
// CUDA Operations //
/////////////////////
__global__ void kset_identity(Matrix *A)
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ((idx / A->GetNumCols()) == (idx % A->GetNumCols()))
  {
    A->GetFlattened()[idx] = 1;
  }
  else
  {
    A->GetFlattened()[idx] = 0;
  }
}

__global__ void ksubtract(Matrix *output, Matrix *A, Matrix *B)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < (A->GetNumCols() * A->GetNumRows())) 
  {
    int row = idx / A->GetNumCols();
    int col = idx % A->GetNumCols();
    (*output)[row][col] = (*A)[row][col] - (*B)[row][col];
  }
}

__global__ void kadd(Matrix *output, Matrix *A, Matrix *B)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < (A->GetNumCols() * A->GetNumRows())) 
  {
    int row = idx / A->GetNumCols();
    int col = idx % A->GetNumCols();
    (*output)[row][col] = (*A)[row][col] + (*B)[row][col];
  }
}

__global__ void kmultiply(Matrix *output, Matrix *input, double scale)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < (input->GetNumCols() * input->GetNumRows())) 
  {
    int row = idx / input->GetNumCols();
    int col = idx % input->GetNumCols();
    (*output)[row][col] = (*input)[row][col] * scale;
  }
}


void Matrix::ToZeroes(void)
{
  int num_blocks = (num_cols * num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kset_zeroes<<<num_blocks, THREADS_PER_BLOCK>>>(this->flat);
  cudaDeviceSynchronize();
}

void Matrix::ToIdentity(void)
{
  int num_blocks = (num_cols * num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kset_identity<<<num_blocks, THREADS_PER_BLOCK>>>(this->flat, this->num_cols);
  cudaDeviceSynchronize();
}

////////////////////////
// Operator Overloads //
////////////////////////
__host__ __device__ double * Matrix::operator[](int row_idx)
{
  return &(this->flat[row_idx * this->num_cols]);
}

Matrix& Matrix::operator-(Matrix& other) 
{
  Matrix *blah = new Matrix(this->num_rows, this->num_cols);
  int num_blocks = (this->num_cols * this->num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  ksubtract<<<num_blocks, THREADS_PER_BLOCK>>>(blah, this, &other);
  return *blah;
}

Matrix& Matrix::operator+(Matrix& other) 
{
  Matrix *blah = new Matrix(this->num_rows, this->num_cols);
  int num_blocks = (this->num_cols * this->num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kadd<<<num_blocks, THREADS_PER_BLOCK>>>(blah, this, &other);
  return *blah;
}

Matrix& Matrix::operator*(double scale) 
{
  Matrix *blah = new Matrix(this->num_rows, this->num_cols);
  int num_blocks = (this->num_cols * this->num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  kmultiply<<<num_blocks, THREADS_PER_BLOCK>>>(blah, this, scale);
  return blah;
}



void Matrix::Parse(const char* file)
{
  std::ifstream matrix(file);

  if (this->flat != 0)
  {
    cudaFree(this->flat);
  }

  matrix >> this->num_rows;
  matrix >> this->num_cols;

  cudaMallocManaged(&flat, sizeof(double) * num_rows * num_cols);
  for (int i = 0; i < this->num_rows; ++i)
  {
    for (int j = 0; j < this->num_cols; ++j)
    {
      matrix >> this->flat[i * this->num_cols + j];
    }
  }
}


__host__ __device__ double & Matrix::At(int row, int col)
{
  return (*this)[row][col];
}

__host__ __device__ double * Matrix::GetFlattened(void)
{
  return this->flat;
}

__host__ __device__ int Matrix::GetNumCols(void)
{
  return this->num_cols;
}

__host__ __device__ int Matrix::GetNumRows(void)
{
  return this->num_rows;
}

__global__ void kset_zeroes(double *A)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
  A[idx] = 0;
}

__global__ void kset_identity(double *A, int cols)
{ 
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
  if ((idx / cols) == (idx % cols))
  {
    A[idx] = 1;
  }
  else
  {
    A[idx] = 0;
  }
}

