#include <fstream>

#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/value.h>

#include "benchmark.hpp"
#include "function.hpp"

extern "C" size_t function(mignificient::Invocation invoc) {

  const char* input_data = reinterpret_cast<const char*>(invoc.payload.data);
  Json::Value input_json;
  Json::Reader reader;
  if (!reader.parse(input_data, input_json)) {
    printf("json error %s", input_data);

    size_t s = snprintf(reinterpret_cast<char*>(invoc.result.data), invoc.result.capacity, "{ \"result\": \"json error: %s\"}", input_data); 
    return s;
  }

  int size = input_json["size"].asInt();
  int iters = input_json["iters"].asInt();
  printf("Input size %d, iters %d\n", size, iters);

  char *d_ptr;
  char *h_ptr;
  auto code = cudaMalloc(&d_ptr, sizeof(char) * size);
  h_ptr = static_cast<char *>(malloc(size));

  std::vector<long long int> results =
      execute_benchmark(h_ptr, d_ptr, size, iters);

  std::ofstream of{"result.txt", std::ios::out};
  for (int i = 0; i < iters; ++i) {
    of << i << "," << results[i] << std::endl;
  }
  of.close();
  printf("Iters %d\n", iters);
  free(h_ptr);
  cudaFree(d_ptr);

  return 0;
}
