
#include <cstddef>
#include <cstdint>
#include <string>
#include <cstdlib>

extern "C" size_t input_generator(uint8_t* payload)
{
  int size = std::stoi(std::getenv("SIZE"));
  int iters = std::stoi(std::getenv("ITERS"));
  int* inputs = reinterpret_cast<int*>(payload);

  inputs[0] = size;
  inputs[1] = iters;

  return sizeof(int)*2;
}
