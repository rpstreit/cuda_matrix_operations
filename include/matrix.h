
#ifndef MATRIX_H
#define MATRIX_H

#include "managed.h"

class Matrix : Managed
{
  private:
    double *flat = 0;
    int num_cols = 0;
    int num_rows = 0;

  public:
    Matrix(const char *file);
    Matrix(const Matrix &copy);
    Matrix(void) = delete;

    ~Matrix(void);

    void Parse(const char *file);
    double * GetFlattened(void);
    int GetNumCols(void);
    int GetNumRows(void);
}

#endif
