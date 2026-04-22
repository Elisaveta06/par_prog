#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <windows.h>

std::pair<std::vector<long long>, int> read(const std::string &path)
{
  std::ifstream file(path);
  std::vector<long long> numbers;

  if (!file)
    return {numbers, 0};

  long long number;
  while (file >> number)
    numbers.push_back(number);

  int N = std::sqrt(numbers.size());
  return {numbers, N};
}

void write_file(const std::string &path_output, const std::vector<long long> &data, size_t N)
{
  std::ofstream file(path_output);
  if (!file.is_open())
  {
    std::cout << "Ошибка при открытии файла." << std::endl;
    return;
  }

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
      file << data[i * N + j] << " ";
    file << "\n";
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    SetConsoleOutputCP(CP_UTF8);
  }

  std::string path1, path2, path_output;
  int N = 0;
  std::vector<long long> A, B, C;

  if (rank == 0)
  {
    std::cout << "Enter input1, input2 and output file names: ";
    std::cin >> path1 >> path2 >> path_output;

    auto res1 = read(path1);
    auto res2 = read(path2);

    A = res1.first;
    N = res1.second;
    B = res2.first;
    int N_2 = res2.second;

    if (N != N_2 || N == 0)
    {
      std::cout << "Ошибка: матрицы разных размеров или файлы не найдены!" << std::endl;
      N = 0;
    }
    else
    {
      C.resize(N * N, 0);
    }
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (N == 0)
  {
    MPI_Finalize();
    return 0;
  }

  if (rank != 0)
  {
    B.resize(N * N);
  }

  MPI_Bcast(B.data(), N * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  int rows_per_proc = N / size;
  int remainder = N % size;
  int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int offset = 0;

  for (int i = 0; i < size; ++i)
  {
    int r = rows_per_proc + (i < remainder ? 1 : 0);
    sendcounts[i] = r * N;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<long long> local_A(local_rows * N);
  std::vector<long long> local_C(local_rows * N, 0);

  MPI_Scatterv(A.empty() ? nullptr : A.data(), sendcounts.data(), displs.data(), MPI_LONG_LONG,
               local_A.data(), local_rows * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  double start_time;
  if (rank == 0)
    start_time = MPI_Wtime();

  for (int i = 0; i < local_rows; ++i)
  {
    for (int k = 0; k < N; ++k)
    {
      for (int j = 0; j < N; ++j)
      {
        local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
      }
    }
  }

  MPI_Gatherv(local_C.data(), local_rows * N, MPI_LONG_LONG,
              C.empty() ? nullptr : C.data(), sendcounts.data(), displs.data(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    double end_time = MPI_Wtime();
    std::cout << "Время выполнения (MPI): " << (end_time - start_time) << " секунд" << std::endl;
    write_file(path_output, C, N);
  }

  MPI_Finalize();
  return 0;
}