
#include <cstring>
#include <iostream>

#include "common.h"
#include "matrix.h"
#include "lu_decomposition.h"
#include "tests.h"

#define VERIFY_KEY "verify"

typedef int (*routine_t)(Matrix *A, ...);

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

enum class Operations : int
{
  MATMUL,
 
  // do not enter anything else after here

  COUNT
};

operation_t ops[static_cast<int>(Operations::COUNT)] =
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
    printf("Using device %d:\n", device);
    printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
        deviceProps.name, (int)deviceProps.totalGlobalMem,
        (int)deviceProps.major, (int)deviceProps.minor,
				(int)deviceProps.clockRate);
  } 

  // iterate through test array and execute matching name
}
