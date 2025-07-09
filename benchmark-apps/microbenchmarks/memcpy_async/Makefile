all: latency

.PHONY: latency
latency:
	nvcc -arch=$(CUDA_ARCH) -O3 -shared -I../../include --compiler-options -fPIC -cudart=shared -ljsoncpp -o latency_func.so latency_func.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -I../../include -cudart=shared -o latency latency.cu

.PHONY: clean
clean:
	rm latency
