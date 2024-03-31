all: latency

.PHONY: latency
latency:
	nvcc -arch=$(CUDA_ARCH) -shared -I../../include --compiler-options -fPIC -cudart=shared -o latency.so latency.cu
	#c++ -L$(CUDA_DIR)/lib64 -I$(CUDA_DIR)/include -lcuda -lcudart -o latency-preinit latency-preinit.cpp

.PHONY: clean
clean:
	rm latency
