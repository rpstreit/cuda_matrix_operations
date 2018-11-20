
#include "common.h"
#include <vector>
__global__ void kcombine(Matrix* matrix, Matrix* inverse, Matrix* dest);
__global__ void find_nonzero(double* matrix, int size, int rowId, double* outId);
__global__ void pivot(int rowId, int k, double* matrix, int size);
__global__ void inverseGJE(int j, int k, double* matrix);
__global__ void fixRow(double *matrix, int size, int rowId);
__global__ void fixCol(double *matrix, int size, int colId);

void combineInverse(Matrix* matrix, Matrix* inverse, Matrix* dest){
    int cols = dest->GetNumCols();
    int rows = dest->GetNumRows();

    //int num_blocks = (rows * cols + 512 -1)/512;

    kcombine<<<rows, cols>>>(matrix, inverse, dest);
    //kcombine<<<num_blocks, 512>>>(matrix, inverse, dest);
    cudaDeviceSynchronize();
}

//Assume the matrix is an n*n matrix
void GJE_inverse(Matrix* matrix){
    int size; int j; int k;
    double *d_row; 
    //double *h_row;
    
    int row = matrix->GetNumRows();
    int col = matrix->GetNumCols();

    Matrix *inverse = new Matrix(row, col);
    matrix_copy(inverse, matrix);
    inverse->ToIdentity();

    Matrix *combination = new Matrix(row, col * 2);
    combineInverse(matrix, inverse, combination);
    matrix_print(combination);

    j = 0;
    size = matrix->GetNumCols();

    //h_row = (double*)malloc(sizeof(double) * size);
    cudaMalloc((void**)&d_row, sizeof(double) * (size));

    double* flat_matrix = combination->GetFlattened();

    while(j < size){
        //cudaMemcpy(d_row, h_row, sizeof(double) * size, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice);
        //find k where matrix[k][j] is not 0
        //find_nonzero(flat_matrix, d_size, d_j, d_row);
        find_nonzero<<<1, size>>>(flat_matrix, size, j, d_row);

        /*k = (int)(reduce(d_row, size, MIN));
        //cudaMemcpy(d_k, k, sizeof(int), cudaMemcpyHostToDevice);
        
        //spawn n threads in 1 block 
        pivot<<<1, size>>>(j,k, flat_matrix, size);

        //spawn n threads in 1 block 
        fixRow<<<1, size>>>(flat_matrix, size, j);

        //spawn n threads in each n blocks
        fixCol<<<size, size>>>(flat_matrix, size, j);*/
        j++;
    }

    //matrix_print(combination);
    delete inverse;
    delete combination;
}

__global__ void kcombine(Matrix* matrix, Matrix* inverse, Matrix* dest){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idx_orig = ((threadIdx.x) + blockIdx.x*((blockDim.x/2)));
    bool end = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;
    
    if(!end){
        if(threadIdx.x < (blockDim.x / 2)) {
            dest->GetFlattened()[idx] = matrix->GetFlattened()[idx_orig];
        }
        else {
            dest->GetFlattened()[idx] = inverse->GetFlattened()[idx_orig+1];
	    //dest->GetFlattened()[idx] = 100;
        }
    }
}

__global__ void find_nonzero(double* matrix, int size, int rowId, double* outId){
    int colId = threadIdx.x;
    int num = matrix[size*rowId+colId];
    if(num != 0.0){
        outId[colId] = (double) colId;
    } else{
        outId[colId] = 0;
    }

}

__global__ void pivot(int rowId, int k, double* matrix, int size){
    int colId = threadIdx.x;
    matrix[size*rowId+colId] = matrix[size*rowId+colId] + matrix[size*k+colId];
    //matrix[j][i] = matrix[j][i] + matrix[k][i];
}

__global__ void fixRow(double *matrix, int size, int rowId){
    __shared__ double Ri[512]; //ith row of the matrix, cap at 512 for max threads
    __shared__ double Aii; //diagonal element for ith row

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    //TODO: check
    Aii = matrix[size*rowId+rowId];
    __syncthreads();
    
    //Divide row by diagonal element
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

__global__ void fixCol(double *matrix, int size, int colId){
    int i = threadIdx.x;
    int j = blockIdx.x;

    __shared__ double col[512] ; //colId col
    __shared__ double AColIdj; //jth element of colId row
    __shared__ double colj[512]; //jth column 

    col[i] = matrix[i * size + colId];
    if(col[i] != 0){
        colj[i] = matrix[i*size+j];
        AColIdj = matrix[colId*size+j];
        if(i != colId){
            colj[i] = colj[i] - AColIdj * col[i];
        }
        matrix[i*size+j] = colj[i];
    }
}
