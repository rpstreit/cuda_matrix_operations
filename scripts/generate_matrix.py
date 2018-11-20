import numpy as np
import argparse
import random

PARSER =  argparse.ArgumentParser()
PARSER.add_argument("path", help="Output file path")
PARSER.add_argument("rows", help="Number of rows", type=int)
PARSER.add_argument("cols", help="Number of columns", type=int)
PARSER.add_argument("-i", "--invertible", help="Make invertible matrix", action='store_true')
PARSER.add_argument("-d", "--identity", help="Make identity matrix. Requires square matrix", action='store_true')
PARSER.add_argument("-u", "--upper", help="Make upper triangular matrix", action='store_true')
PARSER.add_argument("-l", "--lower", help="Make lower triangular matrix", action='store_true')
PARSER.add_argument("-spd", "--spdmat", help="Make Symmetric Positve Definite matrix", action='store_true')

ARGS = PARSER.parse_args()

def get_upper(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.random.random_sample() #* float(np.random.randint(-10000, 10000))
            if i == j:
                while val == 0:
                    val = np.random.random_sample() * float(np.random.randint(-10000, 10000))
            matrix[i, j] = val

    return matrix

def get_lower(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.random.random_sample() * float(np.random.randint(-10000, 10000))
            if i == j:
                while val == 0:
                    val = np.random.random_sample() * float(np.random.randint(-10000, 10000))
            matrix[j, i] = val

    return matrix

def SPD(n):
    A = np.random.rand(n, n)
    A = .5 * np.matmul(A, np.transpose(A))
    A = A + n * np.eye(n, n)
    for i in range(n):
        for j in range(n):
            if(A[i][j] < .0001):
                A[i][j] = 0
    return A

def dump(f, matrix):
    for i in range(len(matrix)):
        f.write("\n")
        for j in range(len(matrix[0])):
            if j == 0:
                f.write(str(matrix[i, j]))
            else:
                f.write(" " + str(matrix[i, j]))



def main():
    f = open(ARGS.path, "w")
    f.write(str(ARGS.rows) + "\n" + str(ARGS.cols))
   
    if (ARGS.invertible):
        assert ARGS.rows == ARGS.cols
        U = get_upper(ARGS.rows)
        L = get_lower(ARGS.rows)
        matrix = np.matmul(U, L)
    elif (ARGS.identity):
        assert ARGS.rows == ARGS.cols
        matrix = np.identity(ARGS.rows)
    elif (ARGS.upper):
        assert ARGS.rows == ARGS.cols
        matrix = get_upper(ARGS.rows)
    elif (ARGS.lower):
        assert ARGS.rows == ARGS.cols
        matrix = get_lower(ARGS.rows)
    elif (ARGS.spdmat):
        assert ARGS.rows == ARGS.cols
        matrix = SPD(ARGS.rows)
    else:
        matrix = np.random.rand(ARGS.rows, ARGS.cols)

    dump(f, matrix)
    f.close()

if __name__ == "__main__":
    main()

