#include "common.h"
#include <vector>
#include <iostream>
#include "matrix_inverse.h"


/**
 * Perform steepest descent algorithm on 
 * Ax = b
 * Extremely inaccurate algorithm
 * @param  A_operator the A (an nxn matrix)
 * @param  b_operator the b (a nx1 vector)
 * @return            an nx1 vector of x
 */
Matrix * steepestDescent(Matrix *A_operator, Matrix *b_operator) {
    std::cout << "Starting steepest descent" << std::endl;
    // First define a max acceptable error
    const double error = .001;
    // matrix_print(A_operator);
    // matrix_print(b_operator);

    // Construct a vector for the return
    double size = b_operator->GetNumRows();
    Matrix *x0 = new Matrix(size, 1); // make a column vector

    Matrix *x_current = x0;
    Matrix *d_vector = new Matrix(size, 1);
    // Matrix *d_vector_transpose = new Matrix(1, size);


    // Allocate all intermediate matrices required
    Matrix * A_xk_1 = new Matrix(size, 1);
    Matrix * A_dk_1 = new Matrix(size, 1);
    // Matrix * A_xk_1_b = new Matrix(size, 1);
    // Matrix * dk_1_A_dk_1 = new Matrix(1, 1);
    Matrix * Axk1_boperator = new Matrix(size, 1);
    Matrix * d_vector_scalar = new Matrix(size, 1);


    // CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrix_multiply_scalar(x0, b_operator, .01);
    // Limit how many cycles we can do in case this blows up
    int count = 0;
    do {

        
        matrix_multiply(A_operator, x_current, A_xk_1);
        // matrix_print(A_xk_1);
        matrix_subtract(A_xk_1, b_operator, Axk1_boperator);
        // matrix_print(Axk1_boperator);
        matrix_multiply_scalar(d_vector, Axk1_boperator, -1);
        // std::cout << "\nD_vector is currently:" << std::endl;
        // matrix_print(d_vector);
        
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
        // std::cout << "\nA_dk_1" << std::endl;
        // matrix_print(A_dk_1);


        double scalar = ((double)dk_1sq / (double)denominator) / (50.0 * sqrt(size));
        // std::cout << "\nScalar is " << scalar << std::endl;
        // std::cout << "Numerator is " << dk_1sq << std::endl;
        // std::cout << "Denominator is " << denominator << std::endl;
        matrix_multiply_scalar(d_vector_scalar, d_vector, scalar);

        matrix_add(x_current, d_vector_scalar, x_current);
        // std::cout << "current X:" << std::endl;
        // matrix_print(x_next);
        // Update x

        // Limit the number of iterations
        if(count++ > 100000) {
            std::cout << "Descent iteration limit reached" << std::endl;
            break;
        }
    } while(norm(d_vector) > error);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Steepest Descent took " << elapsed_time << "ms"<< std::endl;


    delete d_vector;
    delete A_xk_1;
    delete A_dk_1;
    delete Axk1_boperator;
    delete d_vector_scalar;

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
    Matrix *pj_scalar = new Matrix(length, 1);

    for(int k=0; k<length-1; k++) {
        for(int j=0; j<=k; j++) {
            // get Pj transpose
            matrix_transpose(p_vectors[j], pj_t);
            // std::cout << "P_vectors[j] transpose:" << std::endl;
            // matrix_print(pj_t);

            // Get Pj transpose * A
            matrix_multiply(pj_t, A_operator, pj_t_A);
            // std::cout << "P_vectors * A_operator:" << std::endl;
            // matrix_print(pj_t_A);

            // Get (Pj transpose * A) * Pk
            matrix_multiply(pj_t_A, p_vectors[k+1], pj_t_A_pk);
            // std::cout << "P_vectors_j * A_operator * P_vectors_k:" << std::endl;
            // matrix_print(pj_t_A_pk);

            // final value of the numerator
            double numerator = pj_t_A_pk->GetFlattened()[0];
            // std::cout << "Numerator: " << numerator << std::endl;


            // Get (Pj transpose * A) * Pj
            matrix_multiply(pj_t_A, p_vectors[j], pj_t_A_pj);
            // std::cout << "P_vectors_j * A_operator * P_vectors_j:" << std::endl;
            // matrix_print(pj_t_A_pj);
            
            // final value of the denominator
            double denominator = pj_t_A_pj->GetFlattened()[0];
            // std::cout << "Denominator: " << denominator << std::endl;

            double multiplier = numerator / denominator;

            matrix_multiply_scalar(pj_scalar, p_vectors[j], multiplier);
            matrix_subtract(p_vectors[k+1], pj_scalar, p_vectors[k+1]);
        }
    }

    delete pj_t;
    delete pj_t_A;
    delete pj_t_A_pk;
    delete pj_t_A_pj;
    delete pk_pj;
    delete pj_scalar;
    /*std::cout << "A_conjugate vectors" << std::endl;
    for(int i=0; i<length; i++) {
        matrix_print(p_vectors[i]);
    }*/
    return p_vectors;
}


/**
 * Perform Conjugate Direction Algorithm
 * @param  A_operator A nxn matrix
 * @param  b_operator b nx1 vector
 * @return            x nx1 vector of solution
 */
Matrix * conjugateDirection(Matrix * A_operator, Matrix * b_operator) {
    int size = A_operator->GetNumRows();
    
    // Create all required matrices
    Matrix *xk = new Matrix(A_operator->GetNumRows(), 1); // make a column vector
    xk->ToZeroes();
    Matrix * A_xk = new Matrix(size, 1);
    Matrix * gk = new Matrix(size, 1);
    gk->ToZeroes();
    Matrix * pk_t = new Matrix(1, size);
    Matrix * pk_t_gk = new Matrix(1, 1);
    Matrix * A_pk = new Matrix(size, 1);
    Matrix * pk_t_A_pk = new Matrix(1, 1);
    Matrix * A_conj_scalar = new Matrix(size, 1);

    // CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::vector<Matrix *> A_conjugates = constructAConjugates(A_operator);
    // we will have our guess of xk be 0 so that gk = b

    // Limited runtime
    double ak;
    for(int k=0; k < size; k++) {
        // Calculate residue
        matrix_multiply(A_operator, xk, A_xk);
        matrix_subtract(b_operator, A_xk, gk);
        // std::cout << "Residue (gk)" << std::endl;
        // matrix_print(gk);

        matrix_transpose(A_conjugates[k], pk_t);
        matrix_multiply(pk_t, gk, pk_t_gk);
        double numerator = pk_t_gk->GetFlattened()[0];
        // std::cout << "Numerator: " << numerator << std::endl;


        // std::cout << "Debug" << std::endl;
        matrix_multiply(A_operator, A_conjugates[k], A_pk);
        matrix_multiply(pk_t, A_pk, pk_t_A_pk);
        ak = pk_t_A_pk->GetFlattened()[0];
        // std::cout << "Denominator: " << ak << std::endl;



        matrix_multiply_scalar(A_conj_scalar, A_conjugates[k], numerator/ak);
        matrix_add(xk, A_conj_scalar, xk);
        // std::cout << "xk so far: " << std::endl;
        // matrix_print(xk);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Conjugate Direction took " << elapsed_time << "ms"<< std::endl;
    


    // Clear intermediates
    delete A_xk;
    delete gk;
    delete pk_t;
    delete pk_t_gk;
    delete A_pk;
    delete pk_t_A_pk;
    delete A_conj_scalar;

    // clear vector
    for(int i=0; i<A_conjugates.size(); i++) {
        delete A_conjugates[i];
    }

    return xk;
}


Matrix * inverseLinearSolver(Matrix * A_operator, Matrix * b_operator) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Matrix *inverse = new Matrix(*A_operator);
    inverse = GJE_inverse(inverse);
    // matrix_print(inverse);    
    Matrix * solution = new Matrix(*b_operator);
    matrix_multiply(inverse, b_operator, solution);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Inverse solver took " << elapsed_time << "ms"<< std::endl;
    return solution;
}

/*        // Find dk-1
        // dk-1 = -1 * (A * xk-1 - b)
        // matrix_multiply(A, x_current, A_xk_1);
        // matrix_subtract(A_xk_1, b_operator, A_xk_1_b);
        // matrix_multiply_scalar(d_vector, A_xk_1_b, -1);
        std::cout << "A_operator, x0" << std::endl;
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

