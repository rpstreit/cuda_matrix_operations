
#include "matrix.h"

using namespace linalg;

Matrix::Matrix(const char *file)
{
  this->Parse(file);
}

Matrix::Matrix(const Matrix &copy) :
  num_rows(copy.num_rows),
  num_cols(copy.num_cols)
{
  this->base = new double[this->num_rows];
  for (int i = 0; i < this->num_rows; ++i)
  {
    this->base[i] = new double[this->num_cols];
    for (int j = 0; j < this->num_cols; ++j)
    {
      this->base[i][j] = other.base[i][j];
    }
  }
}

Matrix::Matrix(void) { }

Matrix::~Matrix(void)
{ 
  for (int i = 0; i < this->num_rows; ++i)
  {
    delete[] this->base[i];
  }
  delete[] this->base;
}

void Matrix::Parse(const char* file)
{
  ifstream matrix(file);

  if (this-base != 0)
  {
    this->freeBase();
  }

  matrix >> this->num_rows;
  matrix >> this->num_cols;

  this->base = new double[this->num_rows];
  for (int i = 0; i < this->num_rows; ++i)
  {
    this->base[i] = new double[this->num_cols];
    for (int j = 0; j < this->num_cols; ++j)
    {
      matrix >> this->base[i][j];
    }
  }
}

double ** Matrix::GetRaw(void)
{
  return this->base;
}

int Matrix::GetNumCols(void)
{
  return this->num_cols;
}

int Matrix::GetNumRows(void)
{
  return this->num_rows;
}
