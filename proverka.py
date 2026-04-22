import numpy as np

def read(path: str) -> list:
  flat = np.loadtxt(path, dtype=int)
  return flat

def mult(flat_A: list, flat_B: list, n: int) -> list:
  A = flat_A.reshape(n, n)
  B = flat_B.reshape(n, n)

  C = A @ B
  return C

if __name__ == "__main__":
  path_1 = input("Введите название файла с значениями первой матрицы: ")
  path_2 = input("Введите название файла с значениями второй матрицы: ")
  path_cpp = input("Введите название файла с значениями вычисленной матрицы на с++: ")

  flat_A = read(path_1)
  flat_B = read(path_2)

  N_1 = int(np.sqrt(flat_A.size))
  N_2 = int(np.sqrt(flat_B.size))

  if N_1 != N_2:
    exit()
  
  flat_Cpp = read(path_cpp).reshape(N_1, N_1)

  flat_C = mult(flat_A, flat_B, N_1)

  if np.array_equal(flat_C, flat_Cpp):
    print("Матрицы равны")
  else:
    print("Матрицы не равны")