#include <chrono>
#include <iostream>

#include "function.hpp"

extern "C" size_t function(mignificient::Invocation invocation) {
  std::cerr << invocation.payload.size << std::endl;
  float x = 1.0;
  float *d_ptr;
  cudaMalloc(&d_ptr, sizeof(float));
  int iters = 300000;
  auto s = std::chrono::high_resolution_clock::now();

  // for(int i = 0; i < 300000; ++i) {
  cudaMemcpy(d_ptr, &x, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&x, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < iters; ++i) {
    cudaMemcpy(d_ptr, &x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&x, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  }
  auto e = std::chrono::high_resolution_clock::now();
  auto d =
      std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      1000000.0;
  printf("%d %.10f\n", iters, d);
  return 0;
}

// int main() {
//     auto s = std::chrono::high_resolution_clock::now();
//     float x = 1.0;
//     float *d_ptr;
//     cudaMalloc(&d_ptr, sizeof(float));
//
//     for(int i = 0; i < 300000; ++i) {
//       cudaMemcpy(d_ptr, &x, sizeof(float), cudaMemcpyHostToDevice);
//       cudaMemcpy(&x, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
//     }
//     auto e = std::chrono::high_resolution_clock::now();
//     auto d =
//     std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() /
//     1000000.0; printf("%.10f\n", d); return 0;
// }
