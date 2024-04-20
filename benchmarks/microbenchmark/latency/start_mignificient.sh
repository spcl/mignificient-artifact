#!/bin/bash

POLL_TYPE=poll EXECUTOR_TYPE=shmem MANAGER_IP=127.0.0.1 CUDA_BINARY=/scratch/mcopik/gpus/mignificient-artifact/benchmarks/microbenchmark/latency/latency_size.so FUNCTION_NAME=function CONTAINER_NAME=client_0 FUNCTION_FILE=/scratch/mcopik/gpus/mignificient-artifact/benchmarks/microbenchmark/latency/latency_size.so LD_PRELOAD=../../../build/gpuless/libgpuless.so  ../../../build/executor/bin/executor_cpp
