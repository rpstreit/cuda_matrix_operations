
#include <cstdio>
#include "managed.h"
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
