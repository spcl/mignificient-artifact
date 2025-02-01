#!/bin/bash


# Store the application name
APP_NAME="hotspot"
APP="/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/rodinia/bfs/original/bfs /work/serverless/2024/gpus/mignificient-artifact/benchmark-inputs/rodinia/bfs/graph1MW_6.txt"

# Create a directory for output files
OUTPUT_DIR="../../data/function-switching-rtx-4070/function_switching/${APP_NAME}/timesharing"
mkdir -p "$OUTPUT_DIR"

# Function to run the application with parallel
run_parallel() {
    parallel --tag "/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/rodinia/hotspot/original/hotspot 512 2 2 /work/serverless/2024/gpus/mignificient-artifact/benchmark-inputs/rodinia/hotspot/temp_512 /work/serverless/2024/gpus/mignificient-artifact/benchmark-inputs/rodinia/hotspot//power_512 $OUTPUT_DIR/output{#}.out > $OUTPUT_DIR/run_worker{#}.out 2>&1" ::: {1..2}
}

run_parallel
## Run the application with parallel twice
#run_parallel 1
#echo ""
#run_parallel 2

echo "Output files have been created in the $OUTPUT_DIR directory."
