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


Matrix * linearSysSolver::steepestDescent() {
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


Matrix * linearSysSolver::inverse() {

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


    Matrix *pj_t = new Matrix(1, length);
    Matrix *pj_t_A = new Matrix(1, length);
    Matrix pj_t_A_pk = new Matrix(1, 1);
    Matrix pj_t_A_pj = new Matrix(1, 1);

    for(int k=1; i<length; i++) {
        for(j=0; j<k; j++) {
            // get Pj transpose
            matrix_transpose(p_vectors[j], pj_t);
            
            // Get Pj transpose * A
            matrix_multiply(pj_t, A_operator, pj_t_A);

            // Get (Pj transpose * A) * Pk
            matrix_multiply(pj_t_A, p_vectors[k], pj_t_A_pk);

            // final value of the numerator
            double numerator = pj_t_A_pk[0][0];

            // Get (Pj transpose * A) * Pj
            matrix_multiply(pj_t_A, p_vectors[k], pj_t_A_pj);
            
            // final value of the denominator
            double denominator = pj_t_A_pj[0][0];

            double multiplier = numerator / denominator;

            p_vectors[k] = p_vectors[k] - multiplier * p_vectors[j]; 
        }
    }

    delete pj_t;
    delete pj_t_A;
    delete pj_t_A_pk;
    delete pj_t_A_pj;

    return p_vectors;
}
