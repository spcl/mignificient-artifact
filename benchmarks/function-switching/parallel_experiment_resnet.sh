#!/bin/bash


# Store the application name
APP_NAME="resnet"
APP="/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/rodinia/bfs/original/bfs /work/serverless/2024/gpus/mignificient-artifact/benchmark-inputs/rodinia/bfs/graph1MW_6.txt"

# Create a directory for output files
OUTPUT_DIR="../../data/function-switching-rtx-4070/function_switching/${APP_NAME}/timesharing"
mkdir -p "$OUTPUT_DIR"

# Function to run the application with parallel
run_parallel() {
    parallel --tag "python /work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/ml-inference/resnet-50/functions.py > $OUTPUT_DIR/run_worker{#}.out 2>&1" ::: {1..2}
}

run_parallel
## Run the application with parallel twice
#run_parallel 1
#echo ""
#run_parallel 2

echo "Output files have been created in the $OUTPUT_DIR directory."
