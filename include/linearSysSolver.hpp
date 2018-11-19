#ifndef __LINEAR_SYS_SOLVER_HPP
#define __LINEAR_SYS_SOLVER_HPP

#include <vector>
#include <cmath>

class linearSysSolver {
    public:
        Matrix *A_operator;
        Matrix *b_operator;
        linearSysSolver(const char* fileA, const char* fileb);

        ~linearSysSolver();

        /**
         * Performs steepest descent algorithm for solving linear systems
         * The theory is that we turn the matrix into a gradient and find the minimum (solution)
         *     by following the direction that takes us closest to the minimum
         */
        Matrix * steepestDescent();

        /**
         * Performs inverse matrix solver
         * Give Ax = b
         * Do x = (A^-1) * b to solve
         */
        Matrix * inverse();
        
        /**
         * Performs Conjugate direction algorithm
         * 
         */
        Matrix * conjugateDirection();

}


#endif