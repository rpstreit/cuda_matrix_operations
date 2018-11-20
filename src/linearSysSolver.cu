#include "common.h"
#include <vector>
#include <iostream>

/**
 * Perform steepest descent algorithm on 
 * Ax = b
 * @param  A_operator the A (an nxn matrix)
 * @param  b_operator the b (a nx1 vector)
 * @return            an nx1 vector of x
 */
Matrix * steepestDescent(Matrix *A_operator, Matrix *b_operator) {
    // First define a max acceptable error
    const double error = .005;

    // Construct a vector for the return
    double size = b_operator->GetNumRows();
    Matrix *x0 = new Matrix(size, 0); // make a column vector

    Matrix *x_current = x0;
    Matrix *x_next = NULL;
    Matrix *d_vector = NULL;


    // Allocate all intermediate matrices required
    Matrix * A_xk_1 = new Matrix(A_operator->GetNumRows(), 1);
    Matrix * A_dk_1 = new Matrix(A_operator->GetNumRows(), 1);
    Matrix * dk_1_A_dk_1 = new Matrix(1, 1);
    
    // Limit how many cycles we can do in case this blows up
    int count = 0;
    do {
        // Find dk-1
        // dk-1 = -1 * (A * xk-1 - b)
        matrix_multiply(A_operator, x_current, A_xk_1);
        
        if(d_vector) {
            delete d_vector;
        }
        // Note that this creates a new matrix
        d_vector = &((*A_xk_1 - *b_operator) * -1);
        
        // numerator of multiplier
        double dk_1sq = pow(norm(d_vector), 2);

        matrix_multiply(A_operator, d_vector, A_dk_1);
        
        // denominator of multiplier
        double denominator = dot_product(d_vector, A_dk_1);    

        x_next = &(*x_current + (*d_vector * (dk_1sq / denominator)));
        
        // Update x
        delete x_current;
        x_current = x_next;

        // Limit the number of iterations
        if(count++ > 500) {
            std::cout << "Descent iteration limit reached" << std::endl;
            break;
        }
    } while(norm(d_vector) > error);

    delete A_xk_1;
    delete A_dk_1;
    delete dk_1_A_dk_1;
    delete d_vector;

    return x_current;
}

/**
 * Construct n A conjugate vectors
 * @param A_operator    A nxn matrix
 * @return              an std::vector of the conjugate vectors
 */
std::vector<Matrix *> constructAConjugates(Matrix * A_operator) {
    // Assuming b_operator is a column vector
    std::vector<Matrix *> p_vectors;
    int length = A_operator->GetNumRows();

    // Create a linearly independent set
    // The easiest set turns out to be essentially an identity matrix
    for(int i=0; i<length; i++) {
        Matrix *temp = new Matrix(length, 1);
        (*temp)[i][0] = 1;
        p_vectors.push_back(temp);
    }

    // Allocate all intermediate matrices
    Matrix *pj_t = new Matrix(1, length);
    Matrix *pj_t_A = new Matrix(1, length);
    Matrix *pj_t_A_pk = new Matrix(1, 1);
    Matrix *pj_t_A_pj = new Matrix(1, 1);

    for(int k=1; k<length; k++) {
        for(int j=0; j<k; j++) {
            // get Pj transpose
            matrix_transpose(p_vectors[j], pj_t);
            
            // Get Pj transpose * A
            matrix_multiply(pj_t, A_operator, pj_t_A);

            // Get (Pj transpose * A) * Pk
            matrix_multiply(pj_t_A, p_vectors[k], pj_t_A_pk);

            // final value of the numerator
            double numerator = (*pj_t_A_pk)[0][0];

            // Get (Pj transpose * A) * Pj
            matrix_multiply(pj_t_A, p_vectors[k], pj_t_A_pj);
            
            // final value of the denominator
            double denominator = (*pj_t_A_pj)[0][0];

            double multiplier = numerator / denominator;
            // TODO does this delete properly?
            p_vectors[k] = &(*(p_vectors[k]) - (*(p_vectors[j]) * multiplier)); 
        }
    }

    delete pj_t;
    delete pj_t_A;
    delete pj_t_A_pk;
    delete pj_t_A_pj;

    return p_vectors;
}


/**
 * Perform Conjugate Direction Algorithm
 * @param  A_operator A nxn matrix
 * @param  b_operator b nx1 vector
 * @return            x nx1 vector of solution
 */
Matrix * conjugateDirection(Matrix * A_operator, Matrix * b_operator) {
    
    Matrix *x0 = new Matrix(A_operator->GetNumRows(), 0); // make a column vector
    Matrix *xcurrent; // make a column vector
    
    std::vector<Matrix *> A_conjugates = constructAConjugates(A_operator);
    // we will have our guess of x0 be 0 so that r0 = b
    
    // Allocated intermediate matrices
    Matrix * pk_t = new Matrix(1, A_conjugates[0]->GetNumRows());
    Matrix * pk_t_r0 = new Matrix(1, 1);
    Matrix * A_pk = new Matrix(A_operator->GetNumCols(), b_operator->GetNumRows());
    Matrix * pk_t_A_pk = new Matrix(1, 1);

    // Limited runtime
    double ak;
    for(int k=0; k < A_operator->GetNumRows(); k++) {
        matrix_transpose(A_conjugates[k], pk_t);
        matrix_multiply(pk_t, b_operator, pk_t_r0);
        int numerator = (*pk_t_r0)[0][0];

        matrix_multiply(A_operator, A_conjugates[k], A_pk);
        matrix_multiply(pk_t, A_pk, pk_t_A_pk);
        ak = (*pk_t_A_pk)[0][0];
        xcurrent = &(*x0 + (*(A_conjugates[k]) * ak));
        delete x0;
        x0 = xcurrent;
    }

    // Clear intermediates
    delete pk_t;
    delete pk_t_r0;
    delete A_pk;
    delete pk_t_A_pk;

    // clear vector
    for(int i=0; i<A_conjugates.size(); i++) {
        delete A_conjugates[i];
    }

    return x0;
}
