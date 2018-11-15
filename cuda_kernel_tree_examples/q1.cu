
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define THREADS_PER_BLOCK 256

std::vector<int> read_input(void);
void write_results(std::vector<int> const &result, std::string file);

int reduce_min(std::vector<int> const &data);
__global__ void kreduce_min(int *g_in, int *g_out, int length);

std::vector<int> last_digits(std::vector<int> const &data);
__global__ void klast_digits(int *g_in, int *g_out, int length);

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
	std::vector<int> digits = last_digits(data);
	cudaEventRecord(stop, 0);

  write_results(digits, "q1b.txt");

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	std::cout << "LAST_DIGITS, AVG TIME: " << elapsed_time << std::endl;
	for (int i = 0; i < data.size(); ++i)
	{
		std::cout << data[i] << " -> " << digits[i] << std::endl;
	}
	std::cout << std::endl;

	cudaEventRecord(start, 0);
	int result = reduce_min(data);
	cudaEventRecord(stop, 0);

	cudaEventElapsedTime(&elapsed_time, start, stop);

	std::cout << "MIN: " << result << ", AVG TIME: " << elapsed_time << std::endl;	

  std::vector<int> write;
  write.push_back(result);

  write_results(write, "q1a.txt");
}

__global__ void kreduce_min(int *g_in, int *g_out, int length)
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

int reduce_min(std::vector<int> const &data)
{
	int *d_in, *d_out;

	cudaMalloc((void **) &d_in, data.size() * sizeof(int));

	cudaMemcpy(d_in, &data[0], data.size() * sizeof(int), cudaMemcpyHostToDevice);
	
	int num_blocks = (data.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int length = data.size();
	cudaMalloc((void **) &d_out, num_blocks * sizeof(int));

	// pretty much making a tree of kernels since there is no
	// communication between blocks	
	for (;;)
	{
		kreduce_min<<<num_blocks, THREADS_PER_BLOCK>>>(d_in, d_out, length);
		cudaDeviceSynchronize();
		if (num_blocks == 1)
		{
			break;
		}
		else
		{
			// swapping pointers to avoid mem transfers, its hacky I know
			int *temp = d_in;
			d_in = d_out;
			d_out = temp;
			length = num_blocks;
			num_blocks = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		}
	}
	
	int h_out;
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	cudaFree(d_out);

	return h_out;
}

__global__ void klast_digits(int *g_in, int *g_out, int length)
{	
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool past_length = idx < length ? false : true;

	if (!past_length)
	{
		g_out[idx] = g_in[idx] % 10;
	}
}

std::vector<int> last_digits(std::vector<int> const &data)
{	
	int *d_in, *d_out;

	cudaMalloc((void **) &d_in, data.size() * sizeof(int));

	cudaMemcpy(d_in, &data[0], data.size() * sizeof(int), cudaMemcpyHostToDevice);
	
	int num_blocks = (data.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int length = data.size();
	cudaMalloc((void **) &d_out, length * sizeof(int));

	klast_digits<<<num_blocks, THREADS_PER_BLOCK>>>(d_in, d_out, length);
	cudaDeviceSynchronize();

	int h_out[length];
	cudaMemcpy(h_out, d_out, length * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	cudaFree(d_out);

	return std::vector<int>(h_out, h_out + length);
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
