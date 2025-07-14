#!/bin/bash

iters=1000

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${SCRIPT_DIR}/../../benchmark-apps/microbenchmarks/kernel/kernel $iters
mv result.txt $1/kernel.txt
