
#include <cstring>
#include <iostream>

#include "common.h"
#include "matrix.h"
#include "lu_decomposition.h"
#include "tests.h"

#define VERIFY_KEY "verify"

typedef int (*routine_t)(int argc, Matrix **argv);

struct operation_t
{
  char const * name; // name expected from command line
  int num_args; // number of matrices to pass to this operation
  routine_t run; // routine to do operation and print result to 
                   // stdout. Return 0 on success (if there is any
                   // reason to fail)
  routine_t verify; // routine to test operation. On success return 0,
                  // otherwise if the test fails, return something else
};

enum Operations
{
  MATMUL,
 
  // do not enter anything else after here

  COUNT
};

operation_t ops[COUNT] =
{
  {"matmul", 2, matmul_run, matmul_verify},
};

int main(int argc, char **argv)
{  
  if (argc < 2)
  {
    std::cerr << "error: no operation specified!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    std::cerr << "error: no devices supporting CUDA" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  int device = 0;
  cudaSetDevice(device);

  cudaDeviceProp deviceProps;
  if (!cudaGetDeviceProperties(&deviceProps, device))
  {
    std::cout << "Using device " << device << ":" << std::endl;
    std::cout << deviceProps.name << "; global mem: " << deviceProps.totalGlobalMem
      << "; compute v" << deviceProps.major << "." << deviceProps.minor << 
      "; clock: " << deviceProps.clockRate << "kHz" << std::endl;
  } 

  int return_code = 1;

  if (!strcmp(argv[1], "help"))
  {
    std::cout << "matrix_ops <operation> <run|verify> <inputfile1> ..." << std::endl;
    exit(0);
  }

  // iterate through test array and execute matching name
  for (int i = 0; i < COUNT; ++i)
  {
    if (!strcmp(argv[1], ops[i].name))
    {
      if (!strcmp(argv[2], "run"))
      {
        int num_matrices = argc - 3;
        if (num_matrices != ops[i].num_args)
        {
          std::cerr << "error: " << ops[i].name << " requires " << ops[i].num_args << " arguments" << std::endl;
          exit(EXIT_FAILURE);
        }
        Matrix *matrices[num_matrices];
        for (int j = 3; j < argc; ++j)
        {
          matrices[j - 3] = new Matrix(argv[j]);
        }
        return_code = ops[i].run(num_matrices, matrices); 
        for (int j = 3; j < argc; ++j)
        {
          delete matrices[j - 3];
        }
        break;
      }
      else if (!strcmp(argv[2], "verify"))
      {
        int num_matrices = argc - 3;
        if (num_matrices != ops[i].num_args)
        {
          std::cerr << "error: " << ops[i].name << " requires " << ops[i].num_args << " arguments" << std::endl;
          exit(EXIT_FAILURE);
        }
        Matrix *matrices[num_matrices];
        for (int j = 3; j < argc; ++j)
        {
          matrices[j - 3] = new Matrix(argv[j]);
        }
        return_code = ops[i].verify(num_matrices, matrices); 
        for (int j = 3; j < argc; ++j)
        {
          delete matrices[j - 3];
        }
        break;
      }
      else
      {
        std::cerr << "error: " << argv[2] << " is not a valid test option. See help for details" << std::endl;
      }
    }
    else if (i == COUNT - 1)
    {
      std::cerr << "error: " << argv[1] << " is not a supported operation" << std::endl;
    }
  }

  return return_code;
}
