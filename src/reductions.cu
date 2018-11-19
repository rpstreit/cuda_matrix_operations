
#include <tuple>

#include "common.h"

__global__ void kreduce_add(double *g_in, double *g_out, int length);
__global__ void kreduce_product(double *g_in, double *g_out, int length);
__global__ void kreduce_min(double *g_in, double *g_out, int length);
__global__ void kreduce_max(double *g_in, double *g_out, int length, int *idx_in, int *idx_out);

__global__ void kget_counting_array(int *result, int length);

std::tuple<int, double> reduce_maxidx(double *data, int length)
{
	double *d_out;
  int *d_idx1, *d_idx2;
	int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cudaMalloc((void **) &d_out, num_blocks * sizeof(double));
  cudaMalloc((void **) &d_idx1, num_blocks * sizeof(int));
  cudaMalloc((void **) &d_idx2, num_blocks * sizeof(int));

  kget_counting_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_idx1, length);

	// pretty much making a tree of kernels since there is no
	// communication between blocks	
	for (;;)
	{
    kreduce_max<<<num_blocks, THREADS_PER_BLOCK>>>(data, d_out, length, d_idx1, d_idx2);
		cudaDeviceSynchronize();
		if (num_blocks == 1)
		{
			break;
		}
		else
		{
			// swapping pointers to avoid mem transfers
      double *temp = data;
			data = d_out;
			d_out = temp;
			length = num_blocks;
      int *tempp = d_idx1;
      d_idx1 = d_idx2;
      d_idx2 = tempp;
			num_blocks = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		}
	}
	
	double h_out;
  int idx_out;
	cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&idx_out, d_idx2, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_out);
  cudaFree(d_idx1);
  cudaFree(d_idx2);

	return std::make_tuple(idx_out, h_out);
}

// reduce
//
// Computes a specified reduction operation in parallel
//
// Inputs: flattened data array, length of such array, reduction
// operation wished to be performed
// Outputs: double result of reduction
double reduce(double *data, int length, Reduction op_type)
{	
	double *d_out;
	int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cudaMalloc((void **) &d_out, num_blocks * sizeof(double));
	// pretty much making a tree of kernels since there is no
	// communication between blocks	
	for (;;)
	{
    switch (op_type)
    {
      case Reduction::MIN: 
        kreduce_min<<<num_blocks, THREADS_PER_BLOCK>>>(data, d_out, length);
        break;

      case Reduction::MAX:
        kreduce_max<<<num_blocks, THREADS_PER_BLOCK>>>(data, d_out, length, 0, 0);
        break;

      case Reduction::ADD:
        kreduce_add<<<num_blocks, THREADS_PER_BLOCK>>>(data, d_out, length);
        break;

      case Reduction::MUL:
        kreduce_product<<<num_blocks, THREADS_PER_BLOCK>>>(data, d_out, length);
        break;
    }
		cudaDeviceSynchronize();
		if (num_blocks == 1)
		{
			break;
		}
		else
		{
			// swapping pointers to avoid mem transfers
      double *temp = data;
			data = d_out;
			d_out = temp;
			length = num_blocks;
			num_blocks = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		}
	}
	
	int h_out;
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_out);

	return h_out;
}

__global__ void kreduce_add(double *g_in, double *g_out, int length)
{
	__shared__ int s_data[THREADS_PER_BLOCK]; // for speed

 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < length ? false : true;

	if (idx < length)
	{
		s_data[tid] = g_in[idx];
	}
	__syncthreads(); // wait for every thing to get loaded into
										// shared memory for this block

	// reduce in shared memory
	// doing it this way makes better use of warps
	// put in a bunch of checks to not go out of bounds
	if (!past_length)
	{
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (tid < i && (idx + i) < length)
			{
				s_data[tid] = s_data[tid] + s_data[tid + i];
      }

			__syncthreads(); // wait for round to finish
		}	
	}
	
	if (threadIdx.x == 0)
	{
		g_out[blockIdx.x] = s_data[0];
	}
}

__global__ void kreduce_product(double *g_in, double *g_out, int length)
{
	__shared__ int s_data[THREADS_PER_BLOCK]; // for speed

 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < length ? false : true;

	if (idx < length)
	{
		s_data[tid] = g_in[idx];
	}
	__syncthreads(); // wait for every thing to get loaded into
										// shared memory for this block

	// reduce in shared memory
	// doing it this way makes better use of warps
	// put in a bunch of checks to not go out of bounds
	if (!past_length)
	{
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (tid < i && (idx + i) < length)
			{
				s_data[tid] = s_data[tid] * s_data[tid + i];
      }

			__syncthreads(); // wait for round to finish
		}	
	}
	
	if (threadIdx.x == 0)
	{
		g_out[blockIdx.x] = s_data[0];
	}
}

__global__ void kreduce_max(double *g_in, double *g_out, int length, int *idx_in, int *idx_out)
{
	__shared__ int s_data[THREADS_PER_BLOCK * 2]; // for speed

 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < length ? false : true;

	if (idx < length)
	{
		s_data[tid] = g_in[idx];
    s_data[THREADS_PER_BLOCK + tid] = idx_in == 0 ? 0 : idx_in[idx];
	}
	__syncthreads(); // wait for every thing to get loaded into
										// shared memory for this block

	// reduce in shared memory
	// doing it this way makes better use of warps
	// put in a bunch of checks to not go out of bounds
	if (!past_length)
	{
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (tid < i && (idx + i) < length)
			{
        if (s_data[tid] < s_data[tid + i])
        {
          s_data[tid] = s_data[tid + i];
          s_data[THREADS_PER_BLOCK + tid] = s_data[THREADS_PER_BLOCK + tid + i];
        }
			}

			__syncthreads(); // wait for round to finish
		}	
	}
	
	if (threadIdx.x == 0)
	{
		g_out[blockIdx.x] = s_data[0];
    idx_out[blockIdx.x] = s_data[THREADS_PER_BLOCK];
	}
}
__global__ void kreduce_min(double *g_in, double *g_out, int length)
{
	__shared__ int s_data[THREADS_PER_BLOCK]; // for speed

 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = idx < length ? false : true;

	if (idx < length)
	{
		s_data[tid] = g_in[idx];
	}
	__syncthreads(); // wait for every thing to get loaded into
										// shared memory for this block

	// reduce in shared memory
	// doing it this way makes better use of warps
	// put in a bunch of checks to not go out of bounds
	if (!past_length)
	{
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
		{
			if (tid < i && (idx + i) < length)
			{
				s_data[tid] = s_data[tid] < s_data[tid + i] ?
						s_data[tid] : s_data[tid + i];
			}

			__syncthreads(); // wait for round to finish
		}	
	}
	
	if (threadIdx.x == 0)
	{
		g_out[blockIdx.x] = s_data[0];
	}
}

