
#include "common.h"
#include <vector>
__global__ void kcombine(Matrix* matrix, Matrix* inverse, Matrix* dest);
__global__ void kseparate(Matrix* matrix, Matrix* final);
__global__ void find_nonzero(double* matrix, int size, int rowId, double* outId);
__global__ void pivot(int rowId, int k, double* matrix, int size);
__global__ void inverseGJE(int j, int k, double* matrix);
__global__ void fixRow(double *matrix, int size, int rowId);
__global__ void fixCol(double *matrix, int size, int colId);

void combineInverse(Matrix* matrix, Matrix* inverse, Matrix* dest){
    int cols = dest->GetNumCols();
    int rows = dest->GetNumRows();

    kcombine<<<rows, cols>>>(matrix, inverse, dest);
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

    Matrix *inverse = new Matrix(row, col);
    matrix_copy(inverse, matrix);
    inverse->ToIdentity();

    Matrix *combination = new Matrix(row, col * 2);
    combineInverse(matrix, inverse, combination);
    //matrix_print(combination);

    j = 0;
    size = matrix->GetNumCols();

    double* flat_matrix = combination->GetFlattened();

    while(j < size){        
        //spawn n threads in 1 block 
        //std::cout << "\nBefore pivot j = " << j<< std::endl;
        //matrix_print(combination);
        pivot<<<1, size*2>>>(j,j, flat_matrix, matrix->GetNumCols());
        cudaDeviceSynchronize();
        //std::cout << "After pivot" << std::endl;
        //matrix_print(combination);

        //spawn n threads in 1 block 
        fixRow<<<1, size*2>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();
        //std::cout << "After fix row" << std::endl;
        //matrix_print(combination);

        //spawn n threads in each n blocks
        fixCol<<<size*2, size*2>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();
        //std::cout << "After fix col" << std::endl;
        //matrix_print(combination);
        j++;
    }
    getFinalMatrix(combination, matrix);
    //matrix_print(matrix);
    delete inverse;
    delete combination;
    return matrix;
}

__global__ void kcombine(Matrix* matrix, Matrix* inverse, Matrix* dest){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idx_orig = ((threadIdx.x) + blockIdx.x*((blockDim.x/2)));
    //int idx_inv = ((threadIdx.x/2) + blockIdx.x*((blockDim.x/2)));
    int idx_inv = ((threadIdx.x - (blockDim.x/2) - 1) + blockIdx.x*((blockDim.x/2)));
    bool end = idx < dest->GetNumRows() * dest->GetNumCols() ? false : true;
    
    if(!end){
        if(threadIdx.x < (blockDim.x / 2)) {
            dest->GetFlattened()[idx] = matrix->GetFlattened()[idx_orig];
        }
        else {
            dest->GetFlattened()[idx] = inverse->GetFlattened()[idx_inv + 1];
	    //dest->GetFlattened()[idx] = 100;
        }
    }
}

__global__ void kseparate(Matrix* matrix, Matrix* final){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int idx_comb = ((threadIdx.x + (blockDim.x )) + blockIdx.x*((blockDim.x*2)));

    bool end = idx < final->GetNumRows() * final->GetNumCols() ? false : true;
    if(!end){
        //if(threadIdx.x < (blockDim.x / 2)) {
            final->GetFlattened()[idx] = matrix->GetFlattened()[idx_comb];
        //}
    }
}

/**
 * Prevent divide by zero error in fix Row
 */
__global__ void pivot(int rowId, int k, double* matrix, int size){
    int colId = threadIdx.x;
    int double_size = size * 2;
    matrix[double_size*rowId+colId] = matrix[double_size*rowId+colId] + matrix[double_size*k+colId];
    //matrix[j][i] = matrix[j][i] + matrix[k][i];
}

__global__ void fixRow(double *matrix, int orig_size, int rowId){
    __shared__ double Ri[512]; //ith row of the matrix, cap at 512 for max threads
    __shared__ double Aii; //diagonal element for ith row

    int size = orig_size * 2;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    Aii = matrix[size*rowId+rowId];
    __syncthreads();
    
    //Divide row by diagonal element
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

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
