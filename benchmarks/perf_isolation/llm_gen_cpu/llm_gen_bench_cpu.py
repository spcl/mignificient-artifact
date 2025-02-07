import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import numpy as np
import psutil
import multiprocessing as mp
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism

def monitor_peak_memory(pid, stop_flag, peak_memory):
    process = psutil.Process(pid)
    current_peak = 0
    while not stop_flag.is_set():
        try:
            memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            current_peak = max(current_peak, memory_mb)
            time.sleep(0.1)  # Sample every 100ms
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    peak_memory.value = current_peak

def run_generation_task(num_threads, model_name, max_generation_length, num_runs):
    
    torch.set_num_threads(num_threads)
    print(f"Set PyTorch to use {num_threads} threads")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_text = "Please tell me a story of roughly 2000 words."
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_length=max_generation_length)
    
    times = []
    peak_memories = []  # Store peak memory for each run
    
    for i in range(num_runs):
        # Shared memory for peak memory value
        peak_memory = mp.Value('d', 0.0)
        stop_flag = mp.Event()
        
        # Start memory monitoring process
        monitor_process = mp.Process(
            target=monitor_peak_memory,
            args=(mp.current_process().pid, stop_flag, peak_memory)
        )
        monitor_process.start()

        try:
            # Run generation
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_length=max_generation_length)
            end_time = time.time()            
        finally:
            stop_flag.set()
            monitor_process.join()
        
        times.append(end_time - start_time)
        peak_memories.append(peak_memory.value)
        print(f'Iter {i}: {times[-1]:.3f} seconds, Peak Memory: {peak_memory.value:.2f} MB')
    
    return times, peak_memories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, required=True,
                      help='Number of CPU threads to use')
    parser.add_argument('--num_runs', type=int, default=8,
                      help='Number of times to run the test')
    parser.add_argument('--model_name', type=str, default="facebook/opt-125m",
                      help='Model to use for generation')
    parser.add_argument('--max_generation_length', type=int, default=128,
                      help='Maximum length for generation')
    args = parser.parse_args()
    
    print(f"\nRunning LLM generation using {args.num_threads} threads")
    print(f"Model: {args.model_name}")
    print(f"Max generation length: {args.max_generation_length}")
    
    elapsed_times, peak_mems = run_generation_task(
        args.num_threads,
        args.model_name,
        args.max_generation_length,
        args.num_runs
    )
    
    print(f"\nResults for {args.num_threads} threads:")
    print(f"Generation time: {np.mean(elapsed_times):.3f} seconds")
    print(f"Peak memory: {np.mean(peak_mems):.1f} MB")
    
    
    # append results to a file
    if not os.path.isfile("cpu_results.csv"):
        with open("results.csv", "w") as f:
            f.write("model_name,max_generation_length,num_threads,latency,peak_memory\n")
    # Save results
    with open('cpu_results.csv', 'a') as f:
        for runtime, peak_mem in zip(elapsed_times, peak_mems):
            f.write(f"{args.model_name},{args.max_generation_length},{args.num_threads},{runtime},{peak_mem}\n")
