#!/bin/bash

sarus run -t -e POLL_TYPE=poll -e EXECUTOR_TYPE=shmem -e MANAGER_IP=148.187.105.35 -e CUDA_BINARY=/artifact/benchmarks/microbenchmark/latency/latency_size.so -e FUNCTION_NAME=function -e CONTAINER_NAME=client_0 -e FUNCTION_FILE=/artifact/benchmarks/microbenchmark/latency/latency_size.so --mount type=bind,source=/tmp,target=/tmp --mount type=bind,source=$(pwd)/../../../,target=/artifact spcleth/mignificient:executor-sarus bash -c "LD_LIBRARY_PATH=/usr/local/cuda-11.6/compat/ LD_PRELOAD=/build/gpuless/libgpuless.so /build/executor/bin/executor_cpp"

