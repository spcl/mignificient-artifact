
#include <chrono>
#include <iostream>
#include <vector>

#include <cuda.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void empty_kernel() { __nanosleep(1000 * 10); }

std::vector<long long int> execute_benchmark(int iters) {
  std::vector<long long int> results;

  results.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    auto s = std::chrono::high_resolution_clock::now();
    empty_kernel<<<{128, 1, 1}, {256, 1, 1}>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    results.push_back(d);
  }

  return results;
}
