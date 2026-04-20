#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <utility>
#include <chrono>
#include <windows.h>
#include <omp.h>

std::pair<std::vector<std::vector<long long>>, int> read(const std::string &path)
{
  std::ifstream file(path);
  std::vector<std::vector<long long>> data;

  if (!file)
    return {data, 0};

  std::vector<long long> numbers;
  long long number;
  while (file >> number)
    numbers.push_back(number);

  int N = std::sqrt(numbers.size());

  int val_index = 0;
  for (int i = 0; i < N; ++i)
  {
    std::vector<long long> row;

    for (int j = 0; j < N; ++j)
    {
      row.push_back(numbers[val_index]);
      ++val_index;
    }

    data.push_back(row);
  }
  return {data, N};
}

void mult(const std::vector<std::vector<long long>> &data_1, const std::vector<std::vector<long long>> &data_2, std::vector<std::vector<long long>> &data_3, int N)
{
#pragma omp parallel for shared(data_1, data_2, data_3, N) default(none)
  for (int i = 0; i < N; ++i)
  {
    for (int k = 0; k < N; ++k)
    {
      for (int j = 0; j < N; ++j)
      {
        data_3[i][j] += data_1[i][k] * data_2[k][j];
      }
    }
  }
}

void write_file(const std::string &path_output, const std::vector<std::vector<long long>> &data, size_t N)
{
  std::ofstream file(path_output);
  if (!file.is_open())
  {
    std::cout << "Ошибка при открытии файла." << std::endl;
    return;
  }

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      file << data[i][j] << " ";
}

int main()
{
  SetConsoleOutputCP(CP_UTF8);

  std::string path1, path2, path_output;
  std::cout << "Введите название первого и второго файлов для считывания файла, а также название файла для записи результата: ";
  std::cin >> path1 >> path2 >> path_output;

  auto [data_1, N_1] = read(path1);
  auto [data_2, N_2] = read(path2);

  if (N_1 == N_2)
  {
    std::vector<std::vector<long long>> data_3(N_1, std::vector<long long>(N_1, 0));
    std::vector<int> threads({1, 2, 4, 8});

    for (auto thread : threads)
    {
      omp_set_num_threads(thread);
      data_3.assign(N_1, std::vector<long long>(N_1, 0));

      auto start = std::chrono::high_resolution_clock::now();
      mult(data_1, data_2, data_3, N_1);
      auto end = std::chrono::high_resolution_clock::now();

      std::cout << "\nПотоков: " << thread << "\nВремя: " << (end - start).count();
    }
    write_file(path_output, data_3, N_1);
  }
}