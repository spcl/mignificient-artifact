#!/bin/bash

iters=100

for size in 4 128 512 1024 2048 4096 8192 16384 131072 524288 1048576 2097152 5242880; do

  #CUDA_VISIBLE_DEVICES=$1 .latency_size_exec $size $iters
  SIZE=$size ITERS=$iters ../../../build/orchestrator/orchestrator $(pwd)/input_generator.so

  mkdir -p ../../../data/microbenchmark/latency/mignificient_baremetal/${2}
  mv /tmp/result.txt ../../../data/microbenchmark/latency/mignificient_baremetal/${2}/${size}_iters_${iters}.txt

done
