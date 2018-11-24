
#include "common.h"
#include <vector>
__global__ void kcombine(Matrix* matrix, Matrix* identity, Matrix* dest);
__global__ void kseparate(Matrix* matrix, Matrix* final);
__global__ void find_nonzero(double* matrix, int size, int rowId, double* outId);
__global__ void pivot(int rowId, int k, double* matrix, int size);
__global__ void inverseGJE(int j, int k, double* matrix);
__global__ void fixRow(double *matrix, int size, int rowId);
__global__ void fixCol(double *matrix, int size, int colId);

void combineIdentity(Matrix* matrix, Matrix* identity, Matrix* dest){
    int cols = dest->GetNumCols();
    int rows = dest->GetNumRows();

    kcombine<<<rows, cols>>>(matrix, identity, dest);
    cudaDeviceSynchronize();
}

void getFinalMatrix(Matrix* combined, Matrix* final){
    int cols = final->GetNumCols();
    int rows = final->GetNumRows();

    kseparate<<<rows, cols>>>(combined, final);
    cudaDeviceSynchronize();
}

//Assume the matrix is an n*n matrix
Matrix* GJE_inverse(Matrix* matrix){
    int size; int j; 
    
    int row = matrix->GetNumRows();
    int col = matrix->GetNumCols();

    //get ideneity matrix
    Matrix *identity = new Matrix(row, col);
    matrix_copy(identity, matrix);
    identity->ToIdentity();

    //augment origin matrix with identity matrix
    Matrix *combination = new Matrix(row, col * 2);
    combineIdentity(matrix, identity, combination);
    //matrix_print(combination);

    j = 0;
    size = matrix->GetNumCols();

    double* flat_matrix = combination->GetFlattened();

    while(j < size){        
        //prevent divide by 0
        pivot<<<1, size>>>(j,j, flat_matrix, matrix->GetNumCols());
        cudaDeviceSynchronize();

        //row reduction
        fixRow<<<1, size>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();

        //clear column
        fixCol<<<size, size>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();
        j++;
    }
    getFinalMatrix(combination, matrix);
    //matrix_print(matrix);
    delete identity;
    delete combination;
    return matrix;
}

/**
 * Combine original matrix with its identity in cuda 
 */
__global__ void kcombine(Matrix* matrix, Matrix* identity, Matrix* dest){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idx_orig = ((threadIdx.x) + blockIdx.x*((blockDim.x/2))); 
    int idx_inv = ((threadIdx.x - (blockDim.x/2) - 1) + blockIdx.x*((blockDim.x/2)));
    bool end = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;
    
    if(!end){
        if(threadIdx.x < (blockDim.x / 2)) {
            dest->GetFlattened()[idx] = matrix->GetFlattened()[idx_orig];
        }
        else {
            dest->GetFlattened()[idx] = identity->GetFlattened()[idx_inv + 1];
        }
    }
}

/**
 * Separate final matrix from its identity in cuda 
 */
__global__ void kseparate(Matrix* matrix, Matrix* final){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idx_comb = ((threadIdx.x + (blockDim.x )) + blockIdx.x*((blockDim.x*2)));

    bool end = idx < final->GetNumRows() * final->GetNumCols() ? false : true;
    if(!end){
        final->GetFlattened()[idx] = matrix->GetFlattened()[idx_comb];
    }
}

/**
 * Prevent divide by zero error in fix Row
 */
__global__ void pivot(int rowId, int k, double* matrix, int size){
    int colId = threadIdx.x;
    int double_size = size * 2;
    matrix[double_size*rowId+colId] = matrix[double_size*rowId+colId] + matrix[double_size*k+colId];
}

/** 
 * Row reduce a row
 */ 
__global__ void fixRow(double *matrix, int orig_size, int rowId){
    __shared__ double Rowi[512]; //ith row of the matrix, cap at 512 for max threads
    __shared__ double diagonal; //diagonal element for ith row

    int size = orig_size * 2;

    int colId = threadIdx.x;
    Rowi[colId] = matrix[size*rowId + colId];
    diagonal = matrix[size*rowId+rowId];
    __syncthreads();
    
    //Divide row by diagonal element
    Rowi[colId] = Rowi[colId]/diagonal;
    matrix[size*rowId+colId] = Rowi[colId];
}

/** 
 * Clear the column
 */ 
__global__ void fixCol(double *matrix, int orig_size, int colId){
    int i = threadIdx.x; 
    int j = blockIdx.x;

    __shared__ double col[512] ; //colId col
    __shared__ double AColIdj; //jth element of colId row
    __shared__ double colj[512]; //jth column 

    int size = orig_size * 2;

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
