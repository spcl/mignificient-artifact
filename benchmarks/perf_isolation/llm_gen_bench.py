import multiprocessing
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import concurrent.futures
import numpy as np
import argparse
import os
import subprocess
import signal
import atexit

# append LD_LIBRARY_PATH with conda env
os.environ['LD_LIBRARY_PATH'] = f'/home/yyuejian/miniconda3/envs/mig_dev/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

class GPUInstanceManager:
    def __init__(self, mode):
        self.mode = mode
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        self.cleanup()
        exit(0)
        
    def setup_mig(self, num_instances):
        print("Destroying existing MIG instances")
        # sudo nvidia-smi -i 0 -c DEFAULT
        # sudo nvidia-smi -i 0 -mig 1
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'DEFAULT'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '1'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dci'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dgi'], capture_output=True)
        
        gpu_profiles = {2: 9, 3: 14, 4: 15, 7: 19}
        profile = gpu_profiles.get(num_instances)
        if not profile:
            raise ValueError(f"Unsupported number of MIG instances: {num_instances}")
            
        # Create GPU instances
        for i in range(num_instances):
            result = subprocess.run(['sudo', 'nvidia-smi', 'mig', '-cgi', f'{profile}', '-C'], 
                                 capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create GPU instance: {result.stderr}")
        
    def setup_mps(self,num_instances, with_env):
        # Stop any existing MPS daemon
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '0'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'EXCLUSIVE_PROCESS'], capture_output=True)
        
        # Start MPS daemon
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        if with_env:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '43'
        else:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '100'
        subprocess.run(['nvidia-cuda-mps-control', '-d'])
        print("Started MPS")
        
    def cleanup(self):
        if self.mode == 'mig':
            subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dci'])
            subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dgi'])
            print("Destroyed all MIG instances")
        else:
            # subprocess.run(['nvidia-cuda-mps-control', '-d'])
            subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True) 
            print("Stopped MPS")
            subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'DEFAULT'], capture_output=True)
            subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '1'], capture_output=True)

class IsolationTest:
    def __init__(self, uu_id, model_name, max_generation_length):
        self.uu_id = uu_id
        self.model_name = model_name
        self.max_generation_length = max_generation_length
        
    def run_inference(self, num_iterations=4):
        # For MIG mode, set specific GPU
        if 'MIG' in str(self.uu_id):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.uu_id)
            
        device = 'cuda'
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        input_text = "Please tell me a story of roughly 2000 words."
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        latencies = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                model.generate(**inputs, max_length=self.max_generation_length)
            latencies.append(time.time() - start_time)
            
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p99': np.percentile(latencies, 99),
            'uu_id': self.uu_id,
            'latencies': latencies
        }

def run_concurrent_tests(num_instances, gpu_ids, model_name, max_generation_length, mode, with_env):
    tests = []
    for gpu_id in gpu_ids:
        test = IsolationTest(uu_id=gpu_id, model_name=model_name, max_generation_length=max_generation_length)
        tests.append(test)
    
    results = []
    raw_latencies = []  # New list for raw data
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_instances) as executor:
        future_to_test = {executor.submit(test.run_inference): i 
                         for i, test in enumerate(tests)}
        
        for future in concurrent.futures.as_completed(future_to_test):
            instance_id = future_to_test[future]
            result = future.result()
            print(result)
            results.append((instance_id, result))
            raw_latencies.append((instance_id, result['latencies']))  # Save raw latencies
    
    # Save raw data to file
    csv_name = 'raw_latencies_with_env.csv' if with_env else 'raw_latencies_no_env.csv'
    with open(csv_name, 'a') as f:
        if os.path.getsize(csv_name) == 0:
            f.write("model_name,max_length,mode,num_instances,instance_id,iteration,latency\n")
        for instance_id, latencies in raw_latencies:
            for iter_num, latency in enumerate(latencies):
                f.write(f"{args.model_name},{max_generation_length},{args.mode},{num_instances},{instance_id},{iter_num},{latency}\n")

    return results

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mig', 'mps'], required=True)
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument('--max_generation_length', type=int, default=256)
    parser.add_argument('--with_env', type=bool, default=False)
    args = parser.parse_args()
    
    manager = GPUInstanceManager(args.mode)
    try:
        if args.mode == 'mig':
            manager.setup_mig(args.num_instances)
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            gpu_ids = [x for x in re.findall(r'(MIG-[a-f0-9-]+)', result.stdout)]
        else:
            manager.setup_mps(args.num_instances,args.with_env)
            # For MPS, we just use sequential IDs
            gpu_ids = list(range(args.num_instances))
        
        results = run_concurrent_tests(args.num_instances, gpu_ids, args.model_name, args.max_generation_length,args.mode,args.with_env)
        
        print(f"\nResults for {args.mode.upper()} with {args.num_instances} instances:")
        print("-" * 50)
        for instance_id, metrics in results:
            print(f"Instance {instance_id}:")
            print(f"Mean latency: {metrics['mean']:.3f}s")
            print(f"Std deviation: {metrics['std']:.3f}s")
            print(f"P99 latency: {metrics['p99']:.3f}s")
            print()
        
        # append results to a file
        if not os.path.isfile("results.csv"):
            with open("results.csv", "w") as f:
                f.write("model_name,max_generation_length,mode,num_instances,instance_id,time_mean,time_std,time_p99\n")
        with open('results.csv', 'a') as f:
            for instance_id, metrics in results:
                f.write(f"{args.model_name},{args.max_generation_length},{args.mode},{args.num_instances},{instance_id},{metrics['mean']},{metrics['std']},{metrics['p99']}\n")
                
            
    finally:
        manager.cleanup()