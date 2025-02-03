include make.config

CXX := $(CUDA_DIR)/bin/nvcc -cudart shared

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CXX) -O3 $(SRC) -I ../../../include -I ${DEPS_PATH}/include -shared --compiler-options -fPIC -o $(EXE).so -I$(INCLUDE) -L$(CUDA_LIB_DIR) --linker-options "-rpath=${DEPS_PATH}/lib64" -L ${DEPS_PATH}/lib64 -ljsoncpp
	$(CXX) -O3 $(SRC) -I ../../../include -I ${DEPS_PATH}/include main.cpp -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) --linker-options "-rpath=${DEPS_PATH}/lib64" -L ${DEPS_PATH}/lib64 -ljsoncpp

enum: $(SRC)
	$(CXX) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CXX) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CXX) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
