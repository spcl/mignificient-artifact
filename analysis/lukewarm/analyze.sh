#!/bin/bash

# Model evaluation time
cat ../../data/lukewarm-rtx-4070/resnet/sample_*/output_executor_user-0-function-0.log | grep eval | awk '{print $4}' >resnet_eval.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No data" }' resnet_eval.txt >resnet_eval_avg.txt

cat ../../data/lukewarm-rtx-4070/bert/sample_*/output_executor_user-0-function-0.log | grep eval | awk '{print $2}' >bert_eval.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No data" }' bert_eval.txt >bert_eval_avg.txt

# Swap in time
cat ../../data/lukewarm-rtx-4070/bert/sample_*/output_gpuless_user-0-function-0.log | grep SwapIn | awk '{print $5}' >bert_swap.txt
cat ../../data/lukewarm-rtx-4070/resnet/sample_*/output_gpuless_user-0-function-0.log | grep SwapIn | awk '{print $5}' >resnet_swap.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No data" }' bert_swap.txt >bert_swap_avg.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No data" }' resnet_swap.txt >resnet_swap_avg.txt
