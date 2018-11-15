
#ifndef MATRIX_H
#define MATRIX_H

namespace linalg
{
  class Matrix
  {
    private:
      double **base = 0;
      int num_cols = 0;
      int num_rows = 0;
      void freeBase(void);

    public:
      Matrix(const char *file);
      Matrix(const Matrix &copy);
      Matrix(void);

      ~Matrix(void);

      void Parse(const char *file);
      double ** GetRaw(void);
      int GetNumCols(void);
      int GetNumRows(void);
  }
}

#endif
