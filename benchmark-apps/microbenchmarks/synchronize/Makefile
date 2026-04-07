all: synchronize

.PHONY: synchronize
synchronize:
	nvcc -arch=$(CUDA_ARCH) -I ${JSONCPP_DIR}/include -O3 -shared -I../../include --compiler-options -fPIC -cudart=shared -L ${JSONCPP_DIR}/lib64 -ljsoncpp -o synchronize_func.so synchronize_func.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -I../../include -cudart=shared -o synchronize synchronize.cu

.PHONY: clean
clean:
	rm synchronize
