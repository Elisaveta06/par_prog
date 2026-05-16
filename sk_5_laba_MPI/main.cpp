#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 0;
  std::vector<long long> A, B, C;

  if (rank == 0)
  {
    if (argc > 1)
      N = std::atoi(argv[1]);
    else
    {
      std::cout << "Razmer matricy N: ";
      std::cin >> N;
    }

    A.resize(N * N);
    B.resize(N * N);
    C.resize(N * N, 0);

    srand(time(nullptr));
    for (int i = 0; i < N * N; ++i)
    {
      A[i] = rand() % 100;
      B[i] = rand() % 100;
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

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  for (int i = 0; i < local_rows; ++i)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];

  MPI_Gatherv(local_C.data(), local_rows * N, MPI_LONG_LONG,
              C.empty() ? nullptr : C.data(), sendcounts.data(), displs.data(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  if (rank == 0)
  {
    std::cout << "N=" << N << " processes=" << size
              << " time=" << end_time - start_time << " seconds\n";
  }

  MPI_Finalize();
  return 0;
}