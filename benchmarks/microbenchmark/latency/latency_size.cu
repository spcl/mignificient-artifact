#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include "function.hpp"

extern "C" size_t function(mignificient::Invocation invocation) {

  int size = reinterpret_cast<const int *>(invocation.payload.data)[0];
  int iters = reinterpret_cast<const int *>(invocation.payload.data)[1];
  printf("Input size %d, iters %d\n", size, iters);

  char *d_ptr;
  char *h_ptr;
  //char h_ptr;
  auto code = cudaMalloc(&d_ptr, sizeof(char) * size);
  //printf("%s\n", cudaGetErrorString(code));
  //cudaMallocHost(&h_ptr, sizeof(char) * size);
  h_ptr = static_cast<char*>(mignificient_malloc(size));
  //printf("Input size %d, iters %d\n", size, iters);

  std::vector<long long int> results;
  results.reserve(iters);

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

    if(ret1 != cudaSuccess || ret2 != cudaSuccess) {
      std::cerr << "error!" << std::endl;
      abort();
    }
  }

  std::ofstream of{"/tmp/result.txt", std::ios::out};
  for (int i = 0; i < iters; ++i) {
    of << i << "," << results[i] << std::endl;
  }
  of.close();
  printf("Iters %d\n", iters);
  mignificient_free(h_ptr);
  return 0;
}

