#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                           \
  do                                                               \
  {                                                                \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess)                                        \
    {                                                              \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                << " -- " << cudaGetErrorString(err) << std::endl; \
      std::exit(EXIT_FAILURE);                                     \
    }                                                              \
  } while (0)

__global__ void matMulKernel(const long long *A,
                             const long long *B,
                             long long *C,
                             int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N)
  {
    long long sum = 0;
    for (int k = 0; k < N; ++k)
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

std::pair<std::vector<long long>, int> read(const std::string &path)
{
  std::ifstream file(path);
  std::vector<long long> numbers;

  if (!file)
  {
    std::cerr << "Ошибка: не удалось открыть файл " << path << std::endl;
    return {numbers, 0};
  }

  long long number;
  while (file >> number)
    numbers.push_back(number);

  int N = static_cast<int>(std::round(std::sqrt(static_cast<double>(numbers.size()))));
  return {numbers, N};
}

void write_file(const std::string &path_output,
                const std::vector<long long> &data,
                int N)
{
  std::ofstream file(path_output);
  if (!file.is_open())
  {
    std::cerr << "Ошибка при открытии файла для записи." << std::endl;
    return;
  }
  for (int i = 0; i < N * N; ++i)
    file << data[i] << " ";
}

float run_once(const long long *d_A, const long long *d_B, long long *d_C,
               int N, int tile, size_t bytes)
{
  dim3 threads(tile, tile);
  dim3 blocks((N + tile - 1) / tile, (N + tile - 1) / tile);

  CUDA_CHECK(cudaMemset(d_C, 0, bytes));
  matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaMemset(d_C, 0, bytes));
  CUDA_CHECK(cudaEventRecord(ev_start));
  matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  return gpu_ms;
}

int main()
{
  std::string path1, path2, path_output;
  std::cout << "Enter the names of the two input files and the output file: ";
  std::cin >> path1 >> path2 >> path_output;

  auto [h_A, N1] = read(path1);
  auto [h_B, N2] = read(path2);

  if (N1 == 0 || N2 == 0)
  {
    std::cerr << "Ошибка чтения матриц.\n";
    return 1;
  }
  if (N1 != N2)
  {
    std::cerr << "Размеры не совпадают: " << N1 << " vs " << N2 << "\n";
    return 1;
  }

  int N = N1;
  size_t bytes = static_cast<size_t>(N) * N * sizeof(long long);

  std::cout << "Размер матрицы: " << N << "x" << N << "\n\n";

  long long *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  const int tile_sizes[] = {8, 16, 32};
  float best_ms = 1e18f;
  int best_tile = 16;

  std::cout << "Block size | Time (ms)\n";
  std::cout << "-----------+----------\n";

  for (int tile : tile_sizes)
  {
    float ms = run_once(d_A, d_B, d_C, N, tile, bytes);
    std::cout << "  " << tile << "x" << tile << "      | " << ms << " ms\n";
    if (ms < best_ms)
    {
      best_ms = ms;
      best_tile = tile;
    }
  }

  std::cout << "\nЛучший block size: " << best_tile << "x" << best_tile
            << " (" << best_ms << " ms)\n";

  std::vector<long long> h_C(static_cast<size_t>(N) * N);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  write_file(path_output, h_C, N);
  std::cout << "Результат записан в " << path_output << "\n";

  return 0;
}
