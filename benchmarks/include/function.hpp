
#ifndef __MIGNIFICIENT_EXECUTOR_FUNCTION_HPP__
#define __MIGNIFICIENT_EXECUTOR_FUNCTION_HPP__

#include <cstddef>
#include <cstdint>

extern "C" void* mignificient_malloc(size_t size)
{
  //std::cerr << "can't!" << std::endl;
  abort();
}

extern "C" void* mignificient_free(void*)
{
  //std::cerr << "can't!" << std::endl;
  abort();
}

namespace mignificient {

  namespace executor {
    struct Runtime;
  }

  struct InvocationData {
    const uint8_t* data;
    size_t size;
  };

  struct InvocationResultData {
    uint8_t* data;
    size_t size;
    size_t capacity;
  };

  struct Invocation {

    Invocation(executor::Runtime & runtime, InvocationData && payload, InvocationResultData && result):
      runtime(runtime),
      payload(payload),
      result(result)
    {}

    void gpu_yield();

    InvocationData payload;
    InvocationResultData result;

  private:
    executor::Runtime& runtime;
  };

}

#endif
