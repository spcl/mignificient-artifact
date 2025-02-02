
Benchmark implemented by Pengyu (Eric) Zhou.
All results are benchmarked on V100.

All benchmarks are compiled from ```cuda-stream/stream-mps.cu```, it is slightly modified version based on ```stream.cu``` in the same directory. it adds CHECK_LAST_CUDA_ERROR and the kernel function: raiseError that triggers fatal gpu error during kernel execution.

(CUDA_ARCH in ```cuda-stream/Makefile``` was set to ```compute_70``` to compile on V100.)

before running benchmarks, remember to set ```CUDA_VISIBLE_DEVICES```, ```CUDA_MPS_PIPE_DIRECTORY```, and ```CUDA_MPS_LOG_DIRECTORY```.

# MPS error containment
To run evaluation of MPS error containment:
```bash
nvidia-cuda-mps-control -d
sh run.sh
```
where variable ```t``` will be the name of folder created to hold results. The program used in run.sh is ```mps-stream```.

To plot the results, use:
```bash
mps-error-plot.py
```
results used for plotting are available in ```bench-results/test11/``` for no error runs,
```bench-results/test12/``` for runs with 4 errors, and
```bench-results/test13/``` for runs with 8 errors
