
// Please create two seperate functions for each
// operation. One that simply dumps results to 
// command line, and one that does some sort of
// verification (doing the same operation with
// lapack for example and comparing the results).
//
// For both please return 0 on success. This is 
// more important for the verifier function


// Here is a constant to account for floating
// precision loss when checking for equality
#define ERROR 1e-10

int matmul_run(Matrix *A, ...);

int matmul_verify(Matrix *A, ...);
