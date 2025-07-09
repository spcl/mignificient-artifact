#!/bin/bash

iters=1000

../../benchmark-apps/microbenchmarks/kernel/kernel $iters
mv result.txt kernel.txt
