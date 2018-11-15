
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <tuple>

#define THREADS_PER_BLOCK 256

std::vector<int> read_input(void);
void write_results(std::vector<int> const &result, std::string file);

std::vector<int> odd_compact(std::vector<int> const &data);
__global__ void kodd_compact(int *g_in, int *map, int *g_out, int length);
__global__ void kprefix_sum(int *g_in, int *g_out, int length);
__global__ void kdist_sums(int *g_sums, int *g_acc, int length_acc);
__global__ void kmask_odds(int *g_in, int *g_out, int length);
__global__ void kgather_sums(int *g_in, int *g_out, int length);
__global__ void ksat_decr(int *g_in, int *g_out, int length);

int main(int argc, char **argv)
{
  std::vector<int> data = read_input();

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


	// timing just for fun
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	std::vector<int> odds = odd_compact(data);
	cudaEventRecord(stop, 0);

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	std::cout << "ODD NUMBERS, AVG TIME: " << elapsed_time << std::endl;
  std::cout << "IDX\tORIG\tODD" << std::endl;
	for (int i = 0; i < data.size(); ++i)
	{
    std::string odd = i < odds.size() ? std::to_string(odds[i]) : "";
		std::cout << i << ":\t" << data[i] << "\t" << odd << std::endl;
	}
	std::cout << std::endl;

  write_results(odds, "q3.txt");

  return 0;
}

void print_array(std::vector<int> data, std::string title,int *arr, int length)
{
  std::cout << "IDX\tORIG\t" << title << std::endl;
  for (int i = 0; i < length; ++i)
  {
    std::string val = i < length ? std::to_string(arr[i]) : "";
    std::cout << i << "\t" << data[i] << "\t" << val << std::endl;
  }
  std::cout << std::endl;
}

std::vector<int> odd_compact(std::vector<int> const &data)
{
  int *d_data, *map, *swap, *compact, *gather, length;
  int *h_compact;
#ifdef DEBUG
  int *h_swap;
#endif

  cudaMalloc((void **) &d_data, data.size() * sizeof(int));
  cudaMalloc((void **) &gather, data.size() * sizeof(int));
  cudaMalloc((void **) &map, data.size() * sizeof(int));
  cudaMalloc((void **) &swap, data.size() * sizeof(int));

	cudaMemcpy(d_data, &data[0], data.size() * sizeof(int), cudaMemcpyHostToDevice);	
	int num_blocks = (data.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  length = data.size();
  kmask_odds<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, gather, length);
  cudaDeviceSynchronize();

#ifdef DEBUG
  h_swap = (int *) malloc (sizeof(int) * data.size());
  cudaMemcpy(h_swap, gather, data.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  print_array(data, "MASK", h_swap, data.size());
#endif

  // tree of prefix sums to maintain that sweet sweet logarithmic
  // time
  // I'm doing fake recursion because piping all this state into
  // another function sounds like a nightmare
  std::stack< std::tuple<int *, int> > stack; int cnt = 0;
  for (;;)
  {
    kprefix_sum<<<num_blocks, THREADS_PER_BLOCK>>>(gather, swap, length);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaMemcpy(h_swap, swap, length * sizeof(int), cudaMemcpyDeviceToHost);
    print_array(data, "PSUM@" + std::to_string(cnt), h_swap, length);
#endif
    
    if (num_blocks > 1)
    {
      cnt++;
      stack.push(std::make_tuple(swap, length));
      kgather_sums<<<num_blocks,THREADS_PER_BLOCK>>>(swap, gather, length);
      cudaDeviceSynchronize();
      length = num_blocks; 
#ifdef DEBUG
      cudaMemcpy(h_swap, gather, length * sizeof(int), cudaMemcpyDeviceToHost);
      print_array(data, "GSUM@" + std::to_string(cnt), h_swap, length);
#endif
      cudaMalloc((void **) &swap, sizeof(int) * length);    
			num_blocks = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    }
    else
    {
      while (!stack.empty())
      {
        cnt--;
        std::tuple<int *, int> accumulator = stack.top();
        stack.pop();
			  num_blocks = (std::get<1>(accumulator) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kdist_sums<<<num_blocks, THREADS_PER_BLOCK>>>(std::get<0>(accumulator), swap, std::get<1>(accumulator));
        cudaDeviceSynchronize();
#ifdef DEBUG
        cudaMemcpy(h_swap, std::get<0>(accumulator), length * sizeof(int), cudaMemcpyDeviceToHost);
        print_array(data, "DSUM@" + std::to_string(cnt), h_swap, std::get<1>(accumulator));
#endif
        cudaFree(swap);
        swap = std::get<0>(accumulator);
      }
      break;
    }
  }

	num_blocks = (data.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
#ifdef DEBUG
  cudaMemcpy(h_swap, swap, data.size() * sizeof(int), cudaMemcpyDeviceToHost);
  print_array(data, "IPSUM", h_swap, data.size());
#endif
  ksat_decr<<<num_blocks, THREADS_PER_BLOCK>>>(swap, map, data.size());
  cudaDeviceSynchronize();
#ifdef DEBUG
  cudaMemcpy(h_swap, map, data.size() * sizeof(int), cudaMemcpyDeviceToHost);
  print_array(data, "XPSUM", h_swap, data.size());
#endif

  cudaMemcpy(&length, &swap[data.size() - 1], sizeof(int), cudaMemcpyDeviceToHost);
  cudaMalloc((void **) &compact, length * sizeof(int));
  h_compact = (int *)malloc(sizeof(int) * length);

#ifdef DEBUG
  cudaMemcpy(h_swap, d_data, data.size() * sizeof(int), cudaMemcpyDeviceToHost);
  print_array(data, "D_DATA", h_swap, data.size());
#endif
  kodd_compact<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, map, compact, data.size());
  cudaDeviceSynchronize();
  cudaMemcpy(h_compact, compact, length * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int> result = std::vector<int>(h_compact, h_compact + length);

  cudaFree(d_data);
  cudaFree(map);
  cudaFree(swap);
  cudaFree(gather);
  cudaFree(compact);
  free(h_compact);

  return result;
}

__global__ void kdist_sums(int *g_acc, int *g_sums, int length_acc)
{
  int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	
  if (g_idx < length_acc && (g_idx / THREADS_PER_BLOCK) > 0)
  {
    g_acc[g_idx] += g_sums[(g_idx / THREADS_PER_BLOCK) - 1];
  }
  __syncthreads();
}

__global__ void kgather_sums(int *g_in, int *g_out, int length)
{ 
  int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int sum_idx = (g_idx + 1) * THREADS_PER_BLOCK - 1;

  if (sum_idx < length)
  {
    g_out[g_idx] = g_in[sum_idx];
  }
  // what we do for last block (the potentially non power of 2
  // on) doesn't matter - it isn't propagated in the combine step
  __syncthreads();
}

__global__ void kmask_odds(int *g_in, int *g_out, int length)
{
  int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = g_idx < length ? false : true;

  if (!past_length) g_out[g_idx] = g_in[g_idx] % 2;
  __syncthreads();
}

__global__ void ksat_decr(int *g_in, int *g_out, int length)
{
  int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = g_idx < length ? false : true;
  if (!past_length)
  {
    int orig = g_in[g_idx];
    g_out[g_idx] = orig > 0 ? orig - 1 : orig;
  }
  __syncthreads();
}

__global__ void kprefix_sum(int *g_in, int *g_out, int length)
{
  __shared__ int scan[THREADS_PER_BLOCK];
 	
  int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
  int idx;
  bool past_length;

  idx = g_idx;
  past_length = g_idx < length ? false : true;
  if (!past_length) scan[tid] = g_in[g_idx];
  else scan[tid] = 0;
  __syncthreads();

  for (int h = 0; h < (int)log2f(THREADS_PER_BLOCK); ++h)
  {
    if (tid % (int)exp2f(h + 1) == 0)
    {
      scan[tid + (int)exp2f(h + 1) - 1] = scan[tid + (int)exp2f(h) - 1] + scan[tid + (int)exp2f(h + 1) - 1];
    }
    __syncthreads();
  }

  if (tid == 0)  scan[THREADS_PER_BLOCK - 1] = 0;
  __syncthreads();

  for (int h = (int)log2f(THREADS_PER_BLOCK) - 1; h >= 0; --h)
  {
    if (tid % (int)exp2f(h + 1) == 0)
    {
      int left_val = scan[tid + (int)exp2f(h) - 1];
      scan[tid + (int)exp2f(h) - 1] = scan[tid + (int)exp2f(h + 1) - 1];
      scan[tid + (int)exp2f(h + 1) - 1] = left_val + scan[tid + (int)exp2f(h + 1) - 1];
    }
    __syncthreads();
  }

  if (!past_length) g_out[idx] = scan[tid] + g_in[idx]; // make inclusive
  __syncthreads();
}

__global__ void kodd_compact(int *g_in, int *map, int *g_out, int length)
{
  __shared__ int mask[THREADS_PER_BLOCK];

 	int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	bool past_length = g_idx < length ? false : true;

  mask[tid] = g_in[g_idx] % 2;
  __syncthreads();

  if (!past_length && mask[tid])
  {
    g_out[map[g_idx]] = g_in[g_idx];
  }
  __syncthreads();
}

std::vector<int> read_input(void)
{
  std::vector<int> result;
  std::string read, token;
  std::ifstream file ("inp.txt");
  if (file.is_open())
  {
    while(std::getline(file, read))
    {
      std::istringstream ss(read);
      while(std::getline(ss, token, ','))
      {
        result.push_back(std::stoi(token));
      }
    }
  }
  else
  {
    std::cerr << "unable to open inp.txt" << std::endl;
    exit(EXIT_FAILURE);
  }

  file.close();

  return result;
}

void write_results(std::vector<int> const &result, std::string file)
{
  std::ofstream file_;
  file_.open(file);
  int i;
  for (i = 0; i < result.size(); ++i)
  {
    if (i != 0) file_ << ", " << result[i];
    else file_ << result[i];
  }
  file_.close();
}
