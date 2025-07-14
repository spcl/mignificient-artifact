#!/bin/bash

iters=1000

for size in 1 4 128 512 1024 2048 4096 8192 16384 131072 524288 1048576 2097152 5242880; do
  echo "Testing memcpy with size $size bytes"
  ../../benchmark-apps/microbenchmarks/memcpy/latency $size $iters
  mv result.txt $1/memcpy_${size}.txt
done
