include make.config

CXX := $(CUDA_DIR)/bin/nvcc -cudart shared

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CXX) $(SRC) -I ../../../include -shared --compiler-options -fPIC -o $(EXE).so -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -ljsoncpp

enum: $(SRC)
	$(CXX) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CXX) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CXX) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
