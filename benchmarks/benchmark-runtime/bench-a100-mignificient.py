#!/usr/bin/env python

import os
import subprocess
import shutil
import json
import time
import signal

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

# Configuration
device = 0

ld_library_path="/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.6.2-vk2v3pwiid3jg5ffedjh5evex6ezxg4p/lib64"
os_path = os.environ.get('LD_LIBRARY_PATH')
if os_path is not None:
    ld_library_path = f"{ld_library_path}:{os_path}"
os.environ['LD_LIBRARY_PATH'] = ld_library_path
print(os.environ['LD_LIBRARY_PATH'])

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
bench_dir = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, 'benchmark-apps', 'ml-inference')
data_dir = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, 'data', 'benchmark-runtime', 'a100')
REPO_DIR = os.environ.get('REPO_DIR')
BUILD_DIR = os.environ.get('BUILD_DIR')

mig_configs = {
    'nomig': None,  # Full GPU
    #'1g': '19',     # 1g.5gb
    #'2g': '14',     # 2g.10gb
    #'3g': '9',      # 3g.20gb
    #'4g': '5',      # 4g.20gb
    #'7g': '0'       # 7g.40gb
}

benchmark_configs = {
    "vgg19": {"has_sizes": False, "iters": 100, "sizes": None},
    "resnet": {"name": "resnet-50", "has_sizes": False, "iters": 100, "sizes": None},
    "bert": {"has_sizes": False, "name": "BERT-SQuAD", "iters": 100, "sizes": None},
    "alexnet": {"has_sizes": False, "iters": 100, "sizes": None},
}

def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()

def setup_directories():
    """Create necessary directories for results"""
    for benchmark in benchmark_configs.keys():
        for mig in mig_configs.keys():
            dir_path = os.path.join(data_dir, 'mignificient', benchmark, mig)
            os.makedirs(dir_path, exist_ok=True)

def create_benchmark_json(benchmark, mig_name, size=None):
    """Create benchmark.json file for the given benchmark and size"""
    config = benchmark_configs[benchmark]
    
    if 'name' in config:
        cubin_path  = os.path.join(bench_dir, config['name'], f"cubin.txt")
        function_path = os.path.join(bench_dir, config['name'], f"functions.py")
    else:
        function_path = os.path.join(bench_dir, benchmark, f"functions.py")
        cubin_path  = os.path.join(bench_dir, benchmark, f"cubin.txt")
    
    # Create input payload
    if size is not None:
        input_payload = json.dumps({"size": size, "iters": config['iters']})
    else:
        input_payload = json.dumps({"iters": config['iters']})
    
    benchmark_json = {
        "address": "http://127.0.0.1:10000",
        "iterations": 100,
        "parallel-requests": 1,
        "mode": "independent",
        "different-users": True,
        "inputs": [
            {
                "function-language": "python",
                "function": "function",
                "function-path": function_path,
                "cubin-analysis": cubin_path,
                "modules": [],
                "mig-instance": mig_name,
                "gpu-memory": 1024,
                "input-payload": input_payload
            }
        ]
    }
    
    # Write to benchmark.json
    print("Create benchmark.json")
    with open("benchmark.json", "w") as f:
        json.dump(benchmark_json, f, indent=2)
        print(benchmark_json)
        print(os.path.exists("benchmark.json"))

def start_orchestrator():
    """Start the orchestrator process"""
    orchestrator_cmd = f"{BUILD_DIR}/orchestrator/orchestrator orchestrator_config_ault.json tools/devices.json"
    
    with open("orchestrator_output.log", "w") as log_file:
        process = subprocess.Popen(
            orchestrator_cmd.split(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
    
    # Give orchestrator time to start up
    time.sleep(2)
    return process

def kill_orchestrator(process):
    """Kill the orchestrator process"""
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()

def run_invoker(benchmark_json_path, result_csv_path):
    """Run the invoker with the given benchmark config"""
    invoker_cmd = f"{BUILD_DIR}/invoker/bin/invoker {benchmark_json_path} {result_csv_path}"
    return subprocess.run(invoker_cmd.split(), capture_output=True, text=True)

def run_benchmark(gpu_id, mig_id, mig_name, benchmark):
    """Run a complete benchmark for the given configuration"""
    config = benchmark_configs[benchmark]
    dir_path = os.path.join(data_dir, 'mignificient', benchmark, mig_name)
    
    # Set up CUDA_VISIBLE_DEVICES
    if mig_id is None:  # Full GPU
        env_var = f'CUDA_VISIBLE_DEVICES={device}'
    else:
        env_var = f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_id}/{mig_id}/0'
    
    # Set environment variable for this process
    old_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mig_id is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'MIG-GPU-{gpu_id}/{mig_id}/0'
    
    try:
        # Start orchestrator
        print(f"    Starting orchestrator for {benchmark}...")
        orchestrator_process = start_orchestrator()
        
        if config['has_sizes']:
            # Run benchmark for each size
            for size in config['sizes']:
                print(f"      Running {benchmark} with size {size}...")
                
                # Create benchmark.json for this size
                create_benchmark_json(benchmark, mig_name, size)
                
                # Run invoker
                result = run_invoker("benchmark.json", "result.csv")
                if result.returncode != 0:
                    print(f"        Error running invoker for size {size}: {result.stderr}")
                    print(result.stdout)
                    print(result.stderr)
                
                result_filename = f"result_{size}.csv"
                result_path = os.path.join(dir_path, result_filename)
                shutil.move("result.txt", result_path)
                print(f"        Results saved to {result_path}")

                result_path = os.path.join(dir_path, f"invocation_{size}.csv")
                shutil.move("result.csv", result_path)

                shutil.move("benchmark.json", os.path.join(dir_path, f"benchmark_{size}.json"))
        else:
            # Run benchmark without sizes
            print(f"      Running {benchmark}...")
            
            # Create benchmark.json
            create_benchmark_json(benchmark, mig_name)
            
            # Run invoker
            print(os.path.exists("benchmark.json"))
            result = run_invoker("benchmark.json", "result.csv")
            if result.returncode != 0:
                print(f"        Error running invoker: {result.stderr}")
                print(result.stdout)
                print(result.stderr)

            print(os.path.realpath(os.path.curdir), os.path.exists("benchmark.json"))
            print(f"      Done {benchmark}...")
            
            result_path = os.path.join(dir_path)
            shutil.move("result.csv", result_path)

            #result_path = os.path.join(dir_path, "invocation.csv")
            #shutil.move("result.csv", result_path)

            print(f"        Results saved to {result_path}")
            #shutil.move("benchmark.json", os.path.join(dir_path, "benchmark.json"))
        
        # Kill orchestrator
        print(f"    Stopping orchestrator for {benchmark}...")
        kill_orchestrator(orchestrator_process)
        
        # Move orchestrator log
        shutil.move("orchestrator_output.log", dir_path)
        shutil.move("output_executor_user-0-function-0.log", dir_path)
        shutil.move("output_gpuless_user-0-function-0.log", dir_path)
        print(f"        Orchestrator and executor logs saved to {dir_path}")
    
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if old_cuda_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_devices
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

def main():
    setup_directories()
    
    gpu_uuid = get_gpuid(device)
    print(f'GPU {device}: {gpu_uuid}')
    
    for mig_config, profile_id in mig_configs.items():
        print(f"\nRunning benchmarks for configuration: {mig_config}")
        
        if mig_config == 'nomig':
            # Disable MIG
            cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')
            instance_id = None
        else:
            # Enable MIG and configure partition
            cmd_prt(f'sudo nvidia-smi -i {device} -mig 1')
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')  # Clear existing config
            
            # Create GPU instance
            instance_id = cmd(f'sudo nvidia-smi mig -i {device} -cgi {profile_id} -C | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"')
        
        # Run each benchmark
        for benchmark in benchmark_configs.keys():
            print(f"  Running {benchmark}...")
            
            run_benchmark(gpu_uuid, instance_id, mig_config, benchmark)
            
        # Cleanup MIG configuration
        if mig_config != 'nomig':
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')
    
    # Disable MIG at the end
    cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

if __name__ == "__main__":
    main()
