all: kernel


.PHONY: kernel
kernel:
	nvcc -arch=$(CUDA_ARCH) -I ${JSONCPP_DIR}/include -O3 -shared -I../../include --compiler-options -fPIC -cudart=shared -L ${JSONCPP_DIR}/lib64 -ljsoncpp -o kernel_func.so kernel_func.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -I../../include -cudart=shared -o kernel kernel.cu

.PHONY: clean
clean:
	rm kernel
