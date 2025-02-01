
## Dependencies

* CUDA 11.6
* Torch 1.12
* Torchvision 0.13

Please check `ld-preload` inside function configs - if torch has dependency on OpenMP,
it will have to be preloaded.

## Execution

1. Paths to the artifact directory are hardcoded, and they need to be changed in *.json files.

2. To run timesharing with CUDA, execute the following scripts.

```
parallel_experiment_bfs.sh
parallel_experiment_alexnet.sh
parallel_experiment_bert.sh
parallel_experiment_hotspot.sh
parallel_experiment_resnet.sh
parallel_experiment_vgg19.sh
```

3. To run device sharing with MIGnificient, execute the following scripts:

```
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} bfs 0
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} hotspot 0
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} resnet 0
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} vgg19 0
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} alexnet 0
./experiment_local.sh ${REPO_DIR} ${BUILD_DIR} ${DATA_DIR_INSIDE_ARTIFACT} bert 0
```

