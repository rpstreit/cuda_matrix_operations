
#include "managed.h"
class TreeNode : public Managed
{
  private:
    TreeNode *parent;
    int num_cols;
    int num_rows;

  public:
    Tree(void);
    Matrix(const Matrix &copy);
    Matrix(int num_rows, int num_cols, bool identity = false);

    ~Matrix(void);

    void Parse(const char *file);
    
    void ToIdentity(void);
    void ToZeroes(void);
    
    __host__ __device__ double * operator[](int row_idx);
    __host__ __device__ double & At(int row, int col);
    __host__ __device__ double * GetFlattened(void);
    __host__ __device__ int GetNumCols(void);
    __host__ __device__ int GetNumRows(void);

};

#endif
