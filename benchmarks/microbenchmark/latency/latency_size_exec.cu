#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include "function.hpp"


int main(int argc, char ** argv)
{
  int size = std::stoi(argv[1]);
  int iters = std::stoi(argv[2]);
  printf("Input size %d, iters %d\n", size, iters);

  char *d_ptr;
  char *h_ptr;
  cudaMalloc(&d_ptr, sizeof(char) * size);
  cudaMallocHost(&h_ptr, sizeof(char) * size);

  std::vector<long long int> results;
  results.reserve(iters);

  // int iters = 300000;
  cudaMemcpy(d_ptr, &h_ptr, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&h_ptr, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < iters; ++i) {
    auto s = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_ptr, &h_ptr, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&h_ptr, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    results.push_back(d);
  }
  printf("Iters %d\n", iters);

  std::ofstream of{"result.txt", std::ios::out};
  for (int i = 0; i < iters; ++i) {
    of << i << "," << results[i] << std::endl;
  }
  of.close();
  return 0;
}
