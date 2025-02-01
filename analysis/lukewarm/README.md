
1. Run `analyze.sh` to get timing averages for swapping and model loading.

2. To get model size in memory, run

```
cat ../../data/lukewarm-rtx-4070/bert/output_gpuless_user-0-function-0.log | grep 'Total size' | tail
cat ../../data/lukewarm-rtx-4070/resnet/output_gpuless_user-0-function-0.log | grep 'Total size' | tail
```
```
```
