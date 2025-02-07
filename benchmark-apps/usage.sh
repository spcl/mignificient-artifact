#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOME_DIR=$(realpath "$SCRIPT_DIR/..")

GPU="H200"
DEVICE=0

DATE=$(date --iso-8601=seconds)
OUTPUT_DIR="$HOME_DIR/data/mig_size_effect/mig-isolation-bench-$DATE"
mkdir -p "$OUTPUT_DIR"

for benchmark in ml-inference/*/; do
    BENCH_NAME=$(basename "$benchmark")
    OUTPUT_SUBDIR="$OUTPUT_DIR/$BENCH_NAME"
    mkdir -p "$OUTPUT_SUBDIR"
    PROGRAM="$benchmark/functions.py"

    echo "Running benchmark $PROGRAM raw"
    ./watcher/a.out 0 >> "$OUTPUT_SUBDIR/raw_usage.txt" &
    WATCHER_PID=$!
    CUDA_VISIBLE_DEVICES=$DEVICE python3 "$PROGRAM" >> "$OUTPUT_SUBDIR/raw_output.txt"
    kill $WATCHER_PID
    wait $WATCHER_PID

    # MIG profiles and corresponding suffixes
    if [ "$GPU" = "A100" ]; then
        PROFILES=("19" "15" "14" "9" "5" "0")
        SUFFIXES=("1g.5gb" "1g.10gb" "2g" "3g" "4g" "7g")
    elif [ "$GPU" = "H200" ]; then
        PROFILES=("19" "20" "15" "14" "9" "5" "0")
        SUFFIXES=("1g.12gb" "1g.12gb_me" "1g.24gb" "2g.24gb" "3g.48gb" "4g.48gb" "7g.96gb")
    else
        PROFILES=()
        SUFFIXES=()
    fi

    # Get the UUID of the physical GPU
    GPU_UUID=$(nvidia-smi -L | grep "GPU $DEVICE:" | sed -rn 's/.*GPU-([a-f0-9-]*).*/\1/p')

    # Enable MIG mode and remove any existing GPU/Compute instances
    sudo nvidia-smi -i "$DEVICE" -mig 1
    sudo nvidia-smi mig -i "$DEVICE" -dgi

    # Create each MIG instance, run the benchmark, remove the instance
    for i in "${!PROFILES[@]}"; do
        # Grab the instance ID from the creation line
        INSTANCE_ID=$(
            sudo nvidia-smi mig -i "$DEVICE" -cgi "${PROFILES[$i]}" -C \
                | head -n1 \
                | sed -rn 's/.*instance ID[ ]+([0-9]*).*/\1/p'
        )
        
        CUDA_VISIBLE_DEVICES="MIG-GPU-$GPU_UUID/$INSTANCE_ID/0" \
            python3 "$PROGRAM" >> "$OUTPUT_SUBDIR/${SUFFIXES[$i]}_isolated_output.txt"

        # Remove the newly created compute + GPU instance
        sudo nvidia-smi mig -i "$DEVICE" -dci
        sudo nvidia-smi mig -i "$DEVICE" -dgi
    done

    # Turn off MIG mode when done
    sudo nvidia-smi -i "$DEVICE" -mig 0
done
