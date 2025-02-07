#!/usr/bin/env python

import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.path.join(SCRIPT_DIR, os.path.pardir)
FILTER_ID1 = ' | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'
FILTER_IDS = ' | grep "created GPU instance" | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'
GPU = "H200"
DEVICE = 0
N = 1 << 27

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()

def create_partitions(partitions):
    i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi {partitions} -C {FILTER_IDS}')
    ids = i.split('\n')
    for k, v in enumerate(ids):
        ids[k] = v.rstrip()
    return ids

def cmd_async(command):
    return subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE)

def read_process(p):
    output, error = p.communicate()
    return output.decode('ascii')

def read_stream_output(ident, stream_out):
    lines = stream_out.split('\n')
    ls = []
    for l in lines[1:]: # mig specific
        # s = l.split(',')
        # if len(s) != 2:
        #     continue
        ls.append(f'{ident},{l}\n')
    return ls

# Setup
date = cmd('date --iso-8601=seconds')
output_dir = os.path.join(HOME_DIR, 'data', 'mig_size_effect', f'mig-isolation-bench-{date}')
os.makedirs(output_dir)
print(f'Output directory: {output_dir}')
benchmarks = [os.path.join("ml-inference", d) for d in os.listdir("ml-inference") if os.path.isdir(os.path.join("ml-inference", d))]
#benchmarks += [d for d in os.listdir("rodinia") if os.path.isdir(os.path.join("rodinia", d))]

# program = "python3 resnet50-py/run-mig.py"
# program = "bfs/bfs-mig ../../benchmark-inputs/rodinia/bfs/graph1MW_6.txt"
# program = "python3 /test1-mig.py"
# out = []

# Run the benchmarking
for benchmark in benchmarks:
    output_subdir = os.path.join(output_dir, benchmark)
    os.makedirs(output_subdir)
    program = os.path.join(benchmark, 'functions.py')

    print(f'Running benchmark {program} raw')
    p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, 'raw_usage.txt')}")
    r = cmd(f"CUDA_VISIBLE_DEVICES={DEVICE} python3 {program} >> {os.path.join(output_subdir, 'raw_output.txt')}")
    p.kill()
    p.wait()

    if GPU == "A100":
        gpu_uuid = get_gpuid(DEVICE)
        print(f'GPU {DEVICE}: {gpu_uuid}')

        # enable MIG, clear config
        cmd(f'sudo nvidia-smi -i {DEVICE} -mig 1')
        cmd(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 1g 5gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 19 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '1g.5gb_isolated_usage.txt')}")
        r = cmd(f"CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program} >> {os.path.join(output_subdir, '1g.5gb_isolated_output.txt')}")
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 1g 10gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 15 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '1g.10gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 2g, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 14 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '2g_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 3g, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 9 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '3g_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 4g, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 5 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '4g_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 7g, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 0 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '7g_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # disable MIG
        cmd_prt(f'sudo nvidia-smi -i {DEVICE} -mig 0')
    elif GPU == "H200":
        gpu_uuid = get_gpuid(DEVICE)
        print(f'GPU {DEVICE}: {gpu_uuid}')

        # enable MIG, clear config
        cmd(f'sudo nvidia-smi -i {DEVICE} -mig 1')
        cmd(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 1g.12gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 19 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '1g_12gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 1g.12gb+me, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 20 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '1g_12gb_me_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 1g.24gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 15 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '1g_24gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 2g.24gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 14 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '2g_24gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 3g.48gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 9 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '3g_48gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 4g.48gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 5 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '4g_48gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # bandwidth 7g.96gb, in isolation
        i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 0 -C {FILTER_ID1}')
        # p = cmd_async(f"./watcher/a.out 0 >> {os.path.join(output_subdir, '7g_96gb_isolated_usage.txt')}")
        r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 python3 {program}')
        # p.kill()
        # p.wait()
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
        cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

        # disable MIG
        cmd_prt(f'sudo nvidia-smi -i {DEVICE} -mig 0')

# #
# # using all partitions at the same time
# #

# # bandwidth 1g+2g+4g
# ids = create_partitions('19,14,5')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
# p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
# p4 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[2]}/0 {program}')
# out += read_stream_output('1g+2g+4g (1g)', read_process(p1))
# out += read_stream_output('1g+2g+4g (2g)', read_process(p2))
# out += read_stream_output('1g+2g+4g (4g)', read_process(p4))
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

# # bandwidth 3g+3g
# ids = create_partitions('9,9')
# p3_1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
# p3_2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
# out += read_stream_output('3g+3g (3g)', read_process(p3_1))
# out += read_stream_output('3g+3g (3g)', read_process(p3_2))
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

# #
# # bandwidth for compute instances
# #

# # bandwidth 1c
# i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -gi {i} -cci 0')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# out += read_stream_output('1c isolated', read_process(p1))
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

# # bandwidth 3c
# i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -gi {i} -cci 2')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# out += read_stream_output('3c isolated', read_process(p1))
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

# # bandwidth 1c+3c+3c
# i = cmd(f'sudo nvidia-smi mig -i {DEVICE} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -gi {i} -cci 0,2,2')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/1 {program}')
# p3 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/2 {program}')
# out += read_stream_output('1c+3c+3c (1c)', read_process(p1))
# out += read_stream_output('1c+3c+3c (3c)', read_process(p2))
# out += read_stream_output('1c+3c+3c (3c)', read_process(p3))
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {DEVICE} -dgi')

# disable MIG
# cmd_prt(f'sudo nvidia-smi -i {DEVICE} -mig 0')

# write output
# f = open(out_file, 'w')
# for line in out:
#     f.write(line)
# f.close()
