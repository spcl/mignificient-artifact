all: kernel

.PHONY: kernel
kernel:
	nvcc -arch=$(CUDA_ARCH) -O3 -shared -I../../include --compiler-options -fPIC -cudart=shared -ljsoncpp -o kernel_func.so kernel_func.cu
	nvcc -arch=$(CUDA_ARCH) -O3 -I../../include -cudart=shared -o kernel kernel.cu

.PHONY: clean
clean:
	rm kernel
