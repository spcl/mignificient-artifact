#!/bin/bash

iters=1000

../../benchmark-apps/microbenchmarks/synchronize/synchronize $iters
mv result.txt synchronize.txt
