#!/bin/bash

mkdir -p build && cd build

DEPS_PATH=/users/mcopik/projects/2024/mignificient/iceoryx/deps/

pybind11_DIR=/users/mcopik/anaconda3/lib/python3.10/site-packages/pybind11 cmake -DCUDNN_DIR=/users/mcopik/projects/2024/mignificient/cudnn/ -DCMAKE_C_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_STANDARD_LIBRARIES="-L${DEPS_PATH}/lib"  -DCMAKE_BUILD_TYPE=Release ../mignificient

make -j16
