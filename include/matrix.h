
#ifndef MATRIX_H
#define MATRIX_H

#include "managed.h"

class Matrix : Managed
{
  private:
    double *flat = 0;
    int num_cols = 0;
    int num_rows = 0;

    __global__ set_identity(void);

  public:
    Matrix(const char *file);
    Matrix(const Matrix &copy);
    Matrix(int num_rows, int num_cols, bool identity = false);
    Matrix(void) = delete;

    ~Matrix(void);

    void Parse(const char *file);
    
    __host__ __device__ double * operator[](int row_idx); 
    __host__ __device__ double * GetFlattened(void);
    __host__ __device__ int GetNumCols(void);
    __host__ __device__ int GetNumRows(void);
}

#endif
