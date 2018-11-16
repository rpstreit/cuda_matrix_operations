/*
void vectormagnitude() {
    return sqrt((x2-x1)^2 + (y2-y1)^2)
}


void matrixMultiple() {
    
}

void matrixTranspose() {
    
}
*/
#include "../include/matrix.h"
#include <iostream>

#define THREADS_PER_BLOCK 256



////////////////////////////
// CUDA Functions Section //
////////////////////////////

/**
 * Sum of a vector, taken from Dr. Garg's GitHub
 * @param d_out vector output
 * @param d_in  vector input
 */
__global__ void global_reduce_add_kernel(float * d_out, float * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

/**
 * CUDA parallel Dot Product (kinda. It just calculates the vector for adding later)
 * @param product the returned dot product
 * @param vector1 first vector
 * @param vector2 second vector
 * @param size    size of the vectors
 */
__global__ void kindaDotProduct(double *product, double* vector1, double* vector2, double size) {
    int idx = threadIDx.x + blockIdx.x * blockDim.x;
    if(idx < size)
        product[idx] = vector1[idx] * vector2[idx];
}


///////////////////////////////
// General Functions Section //
///////////////////////////////

/**
 * Does the dot product of 2 vectors
 * @param vector1   first vector
 * @param vector2   second vector
 * @return          dot product of the vectors....
 */
double dotProduct(linalg::Matrix& vector1, linalg::Matrix& vector2) {
    // CUDA setup
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
    std::cerr << "error: no devices supporting CUDA" << std::endl;
    exit(EXIT_FAILURE);
    }

    int device = 0;
    cudaSetDevice(device);

    // Error checking
    // One of rows or columns must be 1 to be a vector
    int numRows1 = vector1.getNumRows();
    int numCols1 = vector1.getNumCols();
    int numRows2 = vector2.getNumRows();
    int numCols2 = vector2.getNumCols();
    if((numRows1 != 1 && numCols1 != 1) || (numRows2 != 1 && numCols2 != 1)) {
        std::cout << "Input to dotProduct is not a vector" << std::endl;
        return -1337;
    }
    
    // Find the size of the vectors, assuming they're the same size
    double size;
    if(numRows1 == 1) {
        size = numCols1 * sizeof(double);
    } else {
        size = numRows1 * sizeof(double);
    }
    // TODO vectors must be the same size

    double **base1 = vector1.GetRaw();
    double **base2 = vector2.GetRaw();

    double *d_vector1;
    double *d_vector2;
    double *d_intermediate;
    // Allocate and copy the stuff to the device
    cudaMalloc((void **) &d_vector1, size);
    cudaMalloc((void **) &d_vector2, size);
    cudaMalloc((void **) &d_intermediate, size);
    cudaMemcpy(d_vector1, *base1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, *base2, size, cudaMemcpyHostToDevice);

    // Call CUDA function
    kindaDotProduct<<<(size / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(d_intermediate, d_vector1, d_vector2, size);

    int threads = THREADS_PER_BLOCK;
    int blocks = (size / THREADS_PER_BLOCK);

    // Call CUDA function for summing
    // I reuse d_vector1 because I'm lazy
    global_reduce_add_kernel<<<blocks, threads>>>(d_vector1, d_intermediate);
    
    // Call CUDA function for summing part 2
    threads = blocks;
    blocks = 1;
    global_reduce_add_kernel<<<blocks, threads>>>(d_intermediate, d_vector1);


    double product;
    // Copy over answer
    cudaMemcpy(&product, d_intermediate, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_intermediate);

    // Return product
    return product;
} 



linalg::Matrix transpose(linalg::Matrix& matrix) {
    // CUDA setup
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
    std::cerr << "error: no devices supporting CUDA" << std::endl;
    exit(EXIT_FAILURE);
    }

    int device = 0;
    cudaSetDevice(device);
}

