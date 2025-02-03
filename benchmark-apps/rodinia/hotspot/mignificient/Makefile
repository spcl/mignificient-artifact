include make.config

CC := $(CUDA_DIR)/bin/nvcc -cudart shared

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	#$(CC) -O3 $(KERNEL_DIM) $(SRC) -arch="all-major" -I ../../../include -I ${DEPS_PATH}/include -shared --compiler-options -fPIC -o $(EXE).so -I$(INCLUDE) -L$(CUDA_LIB_DIR) --linker-options "-rpath=${DEPS_PATH}/lib64" -L ${DEPS_PATH}/lib64 -ljsoncpp
	$(CC) -O3 $(KERNEL_DIM) $(SRC) -arch="compute_80" -code sm_80 -I ../../../include -I ${DEPS_PATH}/include -shared --compiler-options -fPIC -o $(EXE).so -I$(INCLUDE) -L$(CUDA_LIB_DIR) --linker-options "-rpath=${DEPS_PATH}/lib64" -L ${DEPS_PATH}/lib64 -ljsoncpp
	$(CC) -O3 $(KERNEL_DIM) $(SRC) main.cpp -I ../../../include -I ${DEPS_PATH}/include -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) --linker-options "-rpath=${DEPS_PATH}/lib64" -L ${DEPS_PATH}/lib64 -ljsoncpp

enum: $(SRC)
	$(CC) -O3 $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
