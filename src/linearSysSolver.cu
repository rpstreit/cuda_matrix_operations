#include "common.h"
#include <vector>
#include <iostream>

/**
 * Perform steepest descent algorithm on 
 * Ax = b
 * Extremely inaccurate algorithm
 * @param  A_operator the A (an nxn matrix)
 * @param  b_operator the b (a nx1 vector)
 * @return            an nx1 vector of x
 */
Matrix * steepestDescent(Matrix *A_operator, Matrix *b_operator) {
    std::cout << "Starting linear descent" << std::endl;
    // First define a max acceptable error
    const double error = .001;
    // matrix_print(A_operator);
    // matrix_print(b_operator);

    // Construct a vector for the return
    double size = b_operator->GetNumRows();
    Matrix *x0 = new Matrix(size, 1); // make a column vector
    matrix_multiply_scalar(x0, b_operator, .01);

    Matrix *x_current = x0;
    Matrix *x_next = new Matrix(size, 1);
    Matrix *d_vector = new Matrix(size, 1);
    // Matrix *d_vector_transpose = new Matrix(1, size);


    // Allocate all intermediate matrices required
    Matrix * A_xk_1 = new Matrix(size, 1);
    Matrix * A_dk_1 = new Matrix(size, 1);
    // Matrix * A_xk_1_b = new Matrix(size, 1);
    // Matrix * dk_1_A_dk_1 = new Matrix(1, 1);
    Matrix * Axk1_boperator = new Matrix(size, 1);
    Matrix * d_vector_scalar = new Matrix(size, 1);

    // Limit how many cycles we can do in case this blows up
    int count = 0;
    do {
        // Find dk-1
        // dk-1 = -1 * (A * xk-1 - b)
        // matrix_multiply(A, x_current, A_xk_1);
        // matrix_subtract(A_xk_1, b_operator, A_xk_1_b);
        // matrix_multiply_scalar(d_vector, A_xk_1_b, -1);
/*        std::cout << "A_operator, x0" << std::endl;
        matrix_print(A_operator);
        matrix_print(x0);
        matrix_multiply(A_operator, x0, A_xk_1);
        std::cout << "b_operator, A_xk_1" << std::endl;
        matrix_print(b_operator);
        matrix_print(A_xk_1);

        matrix_subtract(b_operator, A_xk_1, d_vector);
        std::cout << "d_vector" << std::endl;
        matrix_print(d_vector);

        double rt_r = dot_product(d_vector, d_vector);
        std::cout << "rt_r: " << rt_r << std::endl;
        
        matrix_multiply(A_operator, d_vector, A_dk_1);
        std::cout << "A_dk_1" << std::endl;
        matrix_print(A_dk_1);

        double denom = dot_product(d_vector, A_dk_1);
        std::cout << "denom:" << denom << std::endl;

        if(rt_r < error)
            break;

        double alpha = ((double)rt_r / (double)denom);
        std::cout << "alpha:" << alpha << std::endl;

        matrix_multiply_scalar(d_vector_scalar, d_vector, alpha);
        std::cout << "d_vector_scalar" << std::endl;
        matrix_print(d_vector_scalar);

        matrix_add(x_current, d_vector_scalar, x_next);
        std::cout << "Current guess" << std::endl;
        matrix_print(x_next);

        x_current = x_next;
*/
        
        matrix_multiply(A_operator, x_current, A_xk_1);
        // matrix_print(A_xk_1);
        matrix_subtract(A_xk_1, b_operator, Axk1_boperator);
        // matrix_print(Axk1_boperator);
        matrix_multiply_scalar(d_vector, Axk1_boperator, -1);
        std::cout << "\nD_vector is currently:" << std::endl;
        matrix_print(d_vector);
        
        // numerator of multiplier
        double normal = norm(d_vector);
        double dk_1sq = pow(normal, 2);
        if(normal < error) {
            break;
        }
        // std::cout << "Dk_1sq: " << dk_1sq << std::endl;
        matrix_multiply(A_operator, d_vector, A_dk_1);
        // denominator of multiplier
        double denominator = dot_product(d_vector, A_dk_1);    
        std::cout << "\nA_dk_1" << std::endl;
        matrix_print(A_dk_1);


        double scalar = ((double)dk_1sq / (double)denominator) / (50.0 * size);
        std::cout << "\nScalar is " << scalar << std::endl;
        std::cout << "Numerator is " << dk_1sq << std::endl;
        std::cout << "Denominator is " << denominator << std::endl;
        matrix_multiply_scalar(d_vector_scalar, d_vector, scalar);

        matrix_add(x_current, d_vector_scalar, x_next);
        std::cout << "current X:" << std::endl;
        matrix_print(x_next);
        // Update x
        
        x_current = x_next;

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        // Limit the number of iterations
        if(count++ > 5000) {
            std::cout << "Descent iteration limit reached" << std::endl;
            break;
        }
    } while(norm(d_vector) > error);

    delete x0;
    delete A_xk_1;
    delete A_dk_1;
    // delete dk_1_A_dk_1;
    delete d_vector;
    delete Axk1_boperator;
    
    return x_current;
}

/**
 * Construct n A conjugate vectors
 * @param A_operatA_conjugatesor    A nxn matrix
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
    Matrix *pk_pj = new Matrix(length, 1);
    Matrix *pk_pj_scalar = new Matrix(length, 1);

    for(int k=0; k<length-1; k++) {
        for(int j=0; j<k; j++) {
            // get Pj transpose
            matrix_transpose(p_vectors[j], pj_t);
            
            // Get Pj transpose * A
            matrix_multiply(pj_t, A_operator, pj_t_A);

            // Get (Pj transpose * A) * Pk
            matrix_multiply(pj_t_A, p_vectors[k+1], pj_t_A_pk);

            // final value of the numerator
            double numerator = (*pj_t_A_pk)[0][0];

            // Get (Pj transpose * A) * Pj
            matrix_multiply(pj_t_A, p_vectors[j], pj_t_A_pj);
            
            // final value of the denominator
            double denominator = (*pj_t_A_pj)[0][0];

            double multiplier = numerator / denominator;

            matrix_multiply_scalar(pk_pj_scalar, pk_pj, multiplier);
            matrix_subtract(p_vectors[k+1], pk_pj_scalar, p_vectors[k+1]);
        }
    }

    delete pj_t;
    delete pj_t_A;
    delete pj_t_A_pk;
    delete pj_t_A_pj;
    for(int i=0; i<length; i++) {
        matrix_print(p_vectors[i]);
    }
    return p_vectors;
}


/**
 * Perform Conjugate Direction Algorithm
 * @param  A_operator A nxn matrix
 * @param  b_operator b nx1 vector
 * @return            x nx1 vector of solution
 */
Matrix * conjugateDirection(Matrix * A_operator, Matrix * b_operator) {
    
    Matrix *x0 = new Matrix(A_operator->GetNumRows(), 1); // make a column vector
    x0->GetFlattened()[0] = 1;
    Matrix *xcurrent = new Matrix(A_operator->GetNumRows(), 1); // make a column vector
    int size = A_operator->GetNumRows();
    std::vector<Matrix *> A_conjugates = constructAConjugates(A_operator);
    // we will have our guess of x0 be 0 so that r0 = b
    
    // Allocated intermediate matrices
    Matrix * A_x0 = new Matrix(size, 1);
    Matrix * r0 = new Matrix(size, 1);
    Matrix * pk_t = new Matrix(1, A_conjugates[0]->GetNumRows());
    Matrix * pk_t_r0 = new Matrix(1, 1);
    Matrix * A_pk = new Matrix(A_operator->GetNumCols(), 1);
    Matrix * pk_t_A_pk = new Matrix(1, 1);
    Matrix * A_conj_scalar = new Matrix(b_operator->GetNumRows(), 1);

    // Limited runtime
    double ak;
    for(int k=0; k < A_operator->GetNumRows(); k++) {
        // Calculate residue
        matrix_multiply(A_operator, x0, A_x0);
        matrix_subtract(b_operator, A_x0, r0);

        matrix_transpose(A_conjugates[k], pk_t);
        matrix_multiply(pk_t, r0, pk_t_r0);
        int numerator = (*pk_t_r0)[0][0];

        std::cout << "Debug" << std::endl;
        matrix_multiply(A_operator, A_conjugates[k], A_pk);
        matrix_multiply(pk_t, A_pk, pk_t_A_pk);
        ak = (*pk_t_A_pk)[0][0];

        matrix_multiply_scalar(A_conj_scalar, A_conjugates[k], ak);
        matrix_add(x0, A_conj_scalar, xcurrent);
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
