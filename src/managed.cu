
#include <cstdio>
#include "managed.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
void * Managed::operator new(size_t len)
{
  void *ptr;
  gpuErrchk(cudaMallocManaged(&ptr, len));
  gpuErrchk(cudaDeviceSynchronize());
  return ptr;
}

void Managed::operator delete(void *ptr)
{
  cudaDeviceSynchronize();
  cudaFree(ptr);
}
