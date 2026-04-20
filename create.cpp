#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <iostream>
#include <windows.h>

std::vector<std::vector<int>> create_matrix(size_t N)
{
  std::vector<std::vector<int>> matrix(N, std::vector<int>(N, 0));

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(1, 100000);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      matrix[i][j] = dis(gen);

  return matrix;
}

void write_file(const std::string& path_output,const std::vector<std::vector<int>>&data, size_t N)
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

int main() {
  SetConsoleOutputCP(CP_UTF8);
  size_t N;
  std::string path;
  std::cout << "Введите размер матрицы и путь к файлу для записи значений матрицы: ";
  std::cin >> N >> path;

  auto matrix = create_matrix(N);

  write_file(path, matrix, N);
}