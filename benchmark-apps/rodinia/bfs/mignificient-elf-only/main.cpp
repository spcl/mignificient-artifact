#include <string>
#include <iostream>
#include <chrono>

void BFSGraph(const std::string& file, const std::string& output_result);

int main(int argc, char **argv)
{
  auto begin = std::chrono::high_resolution_clock::now();
  BFSGraph(argv[1], argv[2]);
  auto end = std::chrono::high_resolution_clock::now();

  std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << std::endl;
}
