
#include <fstream>

#include "matrix.h"

Matrix::Matrix(const char *file)
{
  this->Parse(file);
}

Matrix::Matrix(const Matrix &copy) :
  num_rows(copy.num_rows),
  num_cols(copy.num_cols)
{
  this->flat = cudaMallocManaged(&flat, sizeof(double) * copy.num_rows * copy.num_cols);
  for (int i = 0; i < this->num_rows; ++i)
  {
    for (int j = 0; j < this->num_cols; ++j)
    {
      this->flat[i * copy.num_cols + j] = copy.flat[i * copy.num_cols + j];
    }
  }
}

Matrix::~Matrix(void)
{ 
  cudaFree(this->flat);
}

double * operator[](int row_idx)
{
  return &(this-flat[row_idx * this->num_cols]);
}

void Matrix::Parse(const char* file)
{
  ifstream matrix(file);

  if (this->flat != 0)
  {
    cudaFree(this->flat);
  }

  matrix >> this->num_rows;
  matrix >> this->num_cols;

  this->flat = cudaMallocManaged(&flat, sizeof(double) * copy.num_rows * copy.num_cols);
  for (int i = 0; i < this->num_rows; ++i)
  {
    for (int j = 0; j < this->num_cols; ++j)
    {
      matrix >> this->flat[i * this->num_cols + j];
    }
  }
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