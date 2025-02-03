
## Dependencies

* CUDA 11.6
* Torch 1.12
* Torchvision 0.13

Please check `ld-preload` inside function configs - if torch has dependency on OpenMP,
it will have to be preloaded.

## Execution

1. Paths to the artifact directory are hardcoded, and they need to be changed in *.json files.

1a. Uncomment the line printing model evaluation time inside `functions.py` for ResNet and inside `bert.py` for BERT-SQuAD.

2. To run swapping, execute the following scripts. Data will be produced in `../../data/lukewarm-rtx-4070/`.
It will make three requests, where one of them will involve swapping out and swapping in.
The experiment will be repeated ten times.

```
./experiment.sh resnet
./experiment.sh bert
```

3. To create size output, run the following commands.

```
${BUILD_DIR}/orchestrator/orchestrator ${REPO_DIR}/config/orchestrator.json ${REPO_DIR}/logs/devices.json
${BUILD_DIR}/invoker/bin/invoker bert.json result.csv
```

`result.csv` can be ignored. We need file `output_gpuless_user-0-function-0.log` and `output_executor_user-0-function-0.log`,
located inside `${BUILD_DIR}`.

Repeat the same for ResNet:

```
${BUILD_DIR}/orchestrator/orchestrator ${REPO_DIR}/config/orchestrator.json ${REPO_DIR}/logs/devices.json
${BUILD_DIR}/invoker/bin/invoker resnet.json result.csv
```
