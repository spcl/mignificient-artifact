#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "function.hpp"

#include "benchmark.hpp"

int main(int argc, char **argv) {
  int size = std::stoi(argv[1]);
  int iters = std::stoi(argv[2]);
  printf("Input size %d, iters %d\n", size, iters);

  char *d_ptr;
  char *h_ptr;
  auto code = cudaMalloc(&d_ptr, size);
  h_ptr = new char[sizeof(char) * size];

  std::vector<long long int> results =
      execute_benchmark(h_ptr, d_ptr, size, iters);

  printf("Iters %d\n", iters);

  std::ofstream of{"result.txt", std::ios::out};
  for (int i = 0; i < iters; ++i) {
    of << i << "," << results[i] << std::endl;
  }
  of.close();

  delete[] h_ptr;
  cudaFree(d_ptr);

  return 0;
}
