
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda.h>

std::vector<long long int> execute_benchmark(char *h_ptr, char *d_ptr, int size,
                                             int iters) {
  std::vector<long long int> results;

  results.reserve(iters);

  memset(h_ptr, 0, size);

  // int iters = 300000;
  cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < iters; ++i) {
    auto s = std::chrono::high_resolution_clock::now();
    int ret1 = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    int ret2 = cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    results.push_back(d);

    if (ret1 != cudaSuccess || ret2 != cudaSuccess) {
      std::cerr << "error!" << std::endl;
      abort();
    }
  }

  return results;
}
