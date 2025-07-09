#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "function.hpp"

#include "benchmark.hpp"

int main(int argc, char **argv) {
  int iters = std::stoi(argv[1]);
  printf("iters %d\n", iters);

  std::vector<long long int> results = execute_benchmark(iters);

  printf("Iters %d\n", iters);

  std::ofstream of{"result.txt", std::ios::out};
  for (int i = 0; i < iters; ++i) {
    of << i << "," << results[i] << std::endl;
  }
  of.close();

  return 0;
}
