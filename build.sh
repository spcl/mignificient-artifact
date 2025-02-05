#!/bin/bash

mkdir -p build && cd build

DEPS_PATH=/capstor/scratch/cscs/ctianche/clariden/miniconda3/envs/mig_dev/

pybind11_DIR=/capstor/scratch/cscs/ctianche/clariden/miniconda3/envs/mig_dev/lib/python3.10/site-packages/pybind11 cmake -DCUDNN_DIR=/capstor/scratch/cscs/ctianche/clariden/miniconda3/envs/mig_dev/ -DCMAKE_C_FLAGS="-I ${DEPS_PATH}/include" -DSPDLOG_LEVEL_TRACE=ON -DCMAKE_CXX_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_STANDARD_LIBRARIES="-L${DEPS_PATH}/lib"  -DCMAKE_BUILD_TYPE=Release ../mignificient

make -j16
