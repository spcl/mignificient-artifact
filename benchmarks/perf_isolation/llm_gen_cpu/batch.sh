#!/bin/bash


for num_threads in 1 2 3 4; do
    for model_name in facebook/opt-125m facebook/opt-350m facebook/opt-2.7b; do
        for max_generation_length in 32 128 512; do
            python llm_gen_bench_cpu.py --model_name $model_name --num_threads $num_threads --max_generation_length $max_generation_length
        done
    done
done

