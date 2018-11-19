
#ifndef MATRIX_H
#define MATRIX_H

#include "managed.h"

class Matrix : public Managed
{
  private:
    double *flat;
    int num_cols;
    int num_rows;
    
    Matrix(void); // Ghetto way of deleting functions

  public:
    Matrix(const char *file);
    Matrix(const Matrix &copy);
    Matrix(int num_rows, int num_cols, bool identity = false);

    ~Matrix(void);

    void Parse(const char *file);
    
    void ToIdentity(void);
    void ToZeroes(void);

    Matrix * operator-(Matrix *other);
    Matrix * operator+(Matrix *other);
    Matrix * operator*(double scale);
    __host__ __device__ double * operator[](int row_idx);
    __host__ __device__ double & At(int row, int col);
    __host__ __device__ double * GetFlattened(void);
    __host__ __device__ int GetNumCols(void);
    __host__ __device__ int GetNumRows(void);
};

#endif
