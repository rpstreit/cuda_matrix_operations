#include <cstring>
#include <iostream>
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
    double *h_flat;
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
    //size= (int*)malloc(sizeof(int));
    //cudaMemcpy(size, &matrix->GetNumCols(), sizeof(int), cudaMemcpyDeviceToHost);

    //h_row = (double*)malloc(sizeof(double) * size);
    h_flat= (double*)malloc(sizeof(double) * size * size);
    cudaMalloc((void**)&d_row, sizeof(double) * (size));

    double* flat_matrix = combination->GetFlattened();

    while(j < size){
        //cudaMemcpy(d_row, h_row, sizeof(double) * size, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice);
        //find k where matrix[k][j] is not 0
        //find_nonzero(flat_matrix, d_size, d_j, d_row);
        //cudaMemcpy(d_flat, flat_matrix, sizeof(double) * row * col, cudaMemcpyHostToDevice);
        //find_nonzero<<<1, size>>>(flat_matrix, size, j, d_row);
        //find_nonzero<<<1, size>>>(flat_matrix, matrix->GetNumCols(), j, d_row);
        //k = (int)(reduce(d_row, size, MIN));
        //k = j;

        // cudaMemcpy(h_flat, flat_matrix, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
        // k = find_nonzero(h_flat, j, size);
        //cudaMemcpy(d_k, k, sizeof(int), cudaMemcpyHostToDevice);
        
        //spawn n threads in 1 block 
        std::cout << "\nBefore pivot j = " << j<< "k = " << k << std::endl;
        matrix_print(combination);
        pivot<<<1, size*2>>>(j,j, flat_matrix, matrix->GetNumCols());
        cudaDeviceSynchronize();
        std::cout << "After pivot" << std::endl;
        matrix_print(combination);
        
        //pivot<<<1, size>>>(2,3, flat_matrix, 4);

        //spawn n threads in 1 block 
        fixRow<<<1, size*2>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();
        std::cout << "After fix row" << std::endl;
        matrix_print(combination);

        //spawn n threads in each n blocks
        fixCol<<<size*2, size*2>>>(flat_matrix, matrix->GetNumCols(), j);
        cudaDeviceSynchronize();
        std::cout << "After fix col" << std::endl;
        matrix_print(combination);
        j++;
    }

    matrix_print(combination);
    delete inverse;
    delete combination;
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

__global__ void find_nonzero(double* matrix, int size, int rowId, double* outId){
    int colId = threadIdx.x;
    int num = matrix[size*rowId+colId];
    if(num != 0.0){
        outId[colId] = (double) colId;
    } else{
        outId[colId] = 1000;
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
    //TODO: check8
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
