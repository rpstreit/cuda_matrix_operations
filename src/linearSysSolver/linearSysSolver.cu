#include "linearSysSolver.hpp"

#include "matrix.h"
#include <vector>
/**
 * General constructor
 * Pass in A and b such that Ax = b
 * Makes a local copy of the matrices
 */
linearSysSolver::linearSysSolver(const char* fileA, const char* fileb)
{   
    A_operator = new Matrix(fileA);
    b_operator = new Matrix(fileb);
}

/**
 * General destructor
 * Destroys local copy of matrix
 */
linearSysSolver::~linearSysSolver() {
    delete A_operator;
    delete b_operator;
}


Matrix linearSysSolver::steepestDescent() {
    const double error = .005;
    double size = b_operator->GetNumRows();
    Matrix x0 = new Matrix(size, 0); // make a column vector
    Matrix *x_current = x0;

    // Because Bobby killed the default constructor
    Matrix *x_next = x0;
    Matrix *d_vector = x0;

    while(norm(x_current) > error) {
        d_vector = -1 * (dot_product(A, x_current) - b_operator)
        x_next = x_current + (pow(norm(d_vector), 2) / dot_product(d_vector, matrix_multiply(A_operator, d_vector))) * d_vector;
        x_current = x_next;
    }

    return x_current;
}


vector linearSysSolver::inverse() {
    matrix A_inverse = getInverse(A_operator);
    return matrixMultiple(A_inverse, b_operator);
}



std::vector<Matrix *> constructAConjugates() {
    // Assuming b_operator is a column vector
    std::vector<Matrix *> p_vectors;
    int length = A_operator->GetNumRows();
    // Matrix *p0 = new Matrix(length, 1);
    // (*p0)[0][0] = 1;
    // p_vectors.push_back(p0);


    // Create a linearly independent set
    // The easiest set turns out to be essentially an identity matrix
    for(int i=0; i<length; i++) {
        Matrix *temp = new Matrix(length, 1);
        (*temp)[i][0] = 1;
        p_vectors.push_back(temp);
    }

    for(int i=1; i<length; i++) {
        Matrix q()
    }
}
