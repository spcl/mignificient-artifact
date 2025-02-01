all: latency

.PHONY: latency
latency:
	#nvcc -arch=$(CUDA_ARCH) -shared -I../../include --compiler-options -fPIC -cudart=shared -o latency_size.so latency_size.cu
	#nvcc  -arch=$(CUDA_ARCH) -I../../include -cudart=shared -o latency_size_exec latency_size_exec.cu
	#g++ input_generator.cpp -shared -fPIC -o input_generator.so
	#nvcc -arch=$(CUDA_ARCH) -shared -I../../include --compiler-options -fPIC -cudart=shared -o latency_size.so latency_size.cu
	nvcc  -arch=$(CUDA_ARCH) -I../../include -cudart=shared -o kernel_exec kernel_exec.cu
	#g++ input_generator.cpp -shared -fPIC -o input_generator.so

.PHONY: clean
clean:
	rm latency
