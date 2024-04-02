
## prepare

```
module load cuda/11.6.2.lua
CUDA_ARCH=sm_80 CUDA_DIR=/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.6.2-vk2v3pwiid3jg5ffedjh5evex6ezxg4p/ make

sudo nvidia-smi -i 0 -mig 1
```

## small partition

```
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -i 0
sudo nvidia-smi mig -cci -i 0
```

large partition
```
sudo nvidia-smi mig -cgi 0 -i 0
sudo nvidia-smi mig -cci -i 0
```

remove partitions
```
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
```

verify then

```
nvidia-smi -L
```

select one device

## execute - first the native

```
LD_LIBRARY_PATH=/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.6.2-vk2v3pwiid3jg5ffedjh5evex6ezxg4p/lib64 ./run_native.sh MIG-80484cf9-a25b-5712-a17a-d2a9db19f91d small
```

and then later large

## execute - mignificient bare

```
POLL_TYPE=wait MANAGER_TYPE=shmem MANAGER_IP=127.0.0.1 build/gpuless/manager_trace
```

```
POLL_TYPE=wait EXECUTOR_TYPE=shmem MANAGER_IP=148.187.105.35 CUDA_BINARY=/scratch/mcopik/gpus/mignificient-artifact/benchmarks/microbenchmark/latency/latency_size.so FUNCTION_NAME=function CONTAINER_NAME=client_0 FUNCTION_FILE=/scratch/mcopik/gpus/mignificient-artifact/benchmarks/microbenchmark/latency/latency_size.so LD_PRELOAD=../../../build/gpuless/libgpuless.so ../../../build/executor/bin/executor_cpp
```

```
./run_baremetal_mignificient.sh MIG-199325e5-dbe4-5896-b934-96ef47870368 large
```

## execute - mignificient

```
POLL_TYPE=wait MANAGER_TYPE=shmem MANAGER_IP=127.0.0.1 build/gpuless/manager_trace
```

```
sarus run -t -e POLL_TYPE=wait -e EXECUTOR_TYPE=shmem -e MANAGER_IP=148.187.105.35 -e CUDA_BINARY=/artifact/benchmarks/microbenchmark/latency/latency_size.s
o -e FUNCTION_NAME=function -e CONTAINER_NAME=client_0 -e FUNCTION_FILE=/artifact/benchmarks/microbenchmark/latency/latency_size.so --mount type=bind,source=/tmp,target=/tmp --mount type=bind,source=$(pwd)/../../../,target=/artifact spcleth/mignificient:executor-sarus bash -c "LD_LIBRARY_PATH=/usr/local/cuda-11.6/compat/ LD_PRELOAD=/build/gpuless/libgpuless.so /build/executor/bin/executor_cpp"
```

```
./run_sarus_mignificient.sh MIG-199325e5-dbe4-5896-b934-96ef47870368 large
```
