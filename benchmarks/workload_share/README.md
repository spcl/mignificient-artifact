
Benchmark implemented by Pengyu (Eric) Zhou. The benchmark has been executed always on the same node, excluding the overheads of TCP connection to provide
the optimisitc estimation of API remoting overheads.

Results are from a V100 GPU, bare-metal execution. Node is equipped with a 64-core CPU AMD EPYC 7501 @ 2GHz, and 512 GBs of memory

1. Clone https://github.com/spcl/gpuless-thesis/

2. Build each benchmark.

3. Change hardcoded paths in `benchmark-api-remoting-scientific-all.sh` and run it.
