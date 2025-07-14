#!/bin/bash

iters=1000

../../benchmark-apps/microbenchmarks/synchronize/synchronize $iters
mv result.txt $1/synchronize.txt
