#!/usr/bin/env python

import os
import subprocess
import shutil

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
bench_dir = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, 'benchmark-apps', 'microbenchmarks')
data_dir = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, 'data', 'microbenchmarks', 'a100')

mig_configs = {
    'nomig': None,  # Full GPU
    '1g': '19',     # 1g.5gb
    '2g': '14',     # 2g.10gb
    '3g': '9',      # 3g.20gb
    '4g': '5',      # 4g.20gb
    '7g': '0'       # 7g.40gb
}

benchmarks = ['kernel', 'synchronize', 'memcpy_async', 'memcpy']

def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()

def setup_directories():
    """Create necessary directories for results"""
    for benchmark in benchmarks:
        for mig in mig_configs.keys():
            dir_path = os.path.join(data_dir, benchmark, mig)
            os.makedirs(dir_path, exist_ok=True)

def run_benchmark(gpu_id, mig_id, mig_name, benchmark):
    program = os.path.join(SCRIPT_DIR, f'{benchmark}.sh')
    
    dir_path = os.path.join(data_dir, benchmark, mig_name)
    cmd_str = f'{program} {dir_path}'
    
    # Set up CUDA_VISIBLE_DEVICES
    if mig_id is None:  # Full GPU
        env_prefix = f'CUDA_VISIBLE_DEVICES={device}'
    else:
        env_prefix = f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_id}/{mig_id}/0'
    
    # Run the benchmark
    full_cmd = f'{env_prefix} {cmd_str}'
    return cmd(full_cmd)

def main():
    setup_directories()
    
    gpu_uuid = get_gpuid(device)
    print(f'GPU {device}: {gpu_uuid}')
    
    for mig_config, profile_id in mig_configs.items():
        print(f"\nRunning benchmarks for configuration: {mig_config}")
        
        if mig_config == 'nomig':
            # Disable MIG
            cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')
        else:
            # Enable MIG and configure partition
            cmd_prt(f'sudo nvidia-smi -i {device} -mig 1')
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')  # Clear existing config
            
            # Create GPU instance
            instance_id = cmd(f'sudo nvidia-smi mig -i {device} -cgi {profile_id} -C | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"')
        
        # Run each benchmark
        for benchmark in benchmarks:
            print(f"  Running {benchmark}...")
            
            run_benchmark(gpu_uuid, instance_id if mig_config != 'nomig' else None, mig_config, benchmark)
            
        # Cleanup MIG configuration
        if mig_config != 'nomig':
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
            cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')
    
    # Disable MIG at the end
    cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

if __name__ == "__main__":
    main()
