
void lu_decomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P);

void lu_columndecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *Q);

void lu_blockeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P, int r);

void lu_randomizeddecomposition(Matrix *A, Matrix *L, Matrix *U, Matrix *P, Matrix *Q, int l, int k);
