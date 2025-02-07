#!/bin/bash


for mode in mig mps mps_limit_sm; do
    for instances in 2 3 4 7; do
        for model_name in facebook/opt-125m facebook/opt-350m facebook/opt-2.7b; do
            for max_generation_length in 32 128 512; do
                python llm_gen_bench.py --mode $mode --num_instances $instances --model_name $model_name --max_generation_length $max_generation_length
            done
        done
    done
done

