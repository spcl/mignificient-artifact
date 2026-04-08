#!/bin/bash

set -euo pipefail

BASE_URL="http://127.0.0.1:10000"
NUM_ITERATIONS="${1:-10}"
OUTPUT_FILE="${2:-benchmark_results.csv}"

# Get the first container ID from the API
echo "Fetching container list..."
CONTAINER_JSON=$(curl -s -X GET "${BASE_URL}/containers")
USER=$(echo "$CONTAINER_JSON" | jq -r 'keys[0]')
CONTAINER_ID=$(echo "$CONTAINER_JSON" | jq -r ".[\"${USER}\"][0].id")

if [[ -z "$CONTAINER_ID" || "$CONTAINER_ID" == "null" ]]; then
  echo "ERROR: Could not find a container." >&2
  exit 1
fi

echo "Using user=${USER}, container=${CONTAINER_ID}"
echo "Running ${NUM_ITERATIONS} iterations, writing to ${OUTPUT_FILE}"

# CSV header
echo "op_type,time_us,memory_bytes" >"$OUTPUT_FILE"

PAYLOAD="{\"user\": \"${USER}\", \"container\": \"${CONTAINER_ID}\"}"

for i in $(seq 1 "$NUM_ITERATIONS"); do
  echo "Iteration ${i}/${NUM_ITERATIONS}"

  # swap-off
  RESP=$(curl -s -X POST "${BASE_URL}/swap-off" \
    -H "Content-Type: application/json" \
    --data "$PAYLOAD")
  TIME=$(echo "$RESP" | jq -r '.time_us')
  BYTES=$(echo "$RESP" | jq -r '.memory_bytes')
  echo "swapout,${TIME},${BYTES}" >>"$OUTPUT_FILE"

  # swap-in
  RESP=$(curl -s -X POST "${BASE_URL}/swap-in" \
    -H "Content-Type: application/json" \
    --data "$PAYLOAD")
  TIME=$(echo "$RESP" | jq -r '.time_us')
  BYTES=$(echo "$RESP" | jq -r '.memory_bytes')
  echo "swapin,${TIME},${BYTES}" >>"$OUTPUT_FILE"
done

echo "Done. Results written to ${OUTPUT_FILE}"
