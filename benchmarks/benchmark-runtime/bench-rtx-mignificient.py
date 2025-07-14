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

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
bench_dir = os.path.join(
    SCRIPT_DIR, os.path.pardir, os.path.pardir, "benchmark-apps", "ml-inference"
)
data_dir = os.path.join(
    SCRIPT_DIR, os.path.pardir, os.path.pardir, "data", "benchmark-runtime", "rtx"
)
REPO_DIR = os.environ.get("REPO_DIR")
BUILD_DIR = os.environ.get("BUILD_DIR")

mig_configs = {"nomig": None}

benchmark_configs = {
    "vgg19": {"has_sizes": False, "iters": 100, "sizes": None},
    "resnet": {"has_sizes": False, "iters": 100, "sizes": None},
    "bert": {"has_sizes": False, "iters": 100, "sizes": None},
    "alexnet": {"has_sizes": False, "iters": 100, "sizes": None},
}


def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode("ascii").rstrip()


def setup_directories():
    """Create necessary directories for results"""
    for benchmark in benchmark_configs.keys():
        for mig in mig_configs.keys():
            dir_path = os.path.join(data_dir, "mignificient", benchmark, mig)
            os.makedirs(dir_path, exist_ok=True)


def start_orchestrator():
    """Start the orchestrator process"""
    orchestrator_cmd = f"{BUILD_DIR}/orchestrator/orchestrator orchestrator_config.json tools/devices.json"
    print(orchestrator_cmd)

    with open("orchestrator_output.log", "w") as log_file:
        process = subprocess.Popen(
            orchestrator_cmd.split(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
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
    invoker_cmd = (
        f"{BUILD_DIR}/invoker/bin/invoker {benchmark_json_path} {result_csv_path}"
    )
    print(invoker_cmd)
    return subprocess.run(invoker_cmd.split(), capture_output=True, text=True)


def run_benchmark(gpu_id, mig_id, mig_name, benchmark):
    """Run a complete benchmark for the given configuration"""
    config = benchmark_configs[benchmark]
    dir_path = os.path.join(data_dir, "mignificient", benchmark, mig_name)

    # Set environment variable for this process
    old_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mig_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"MIG-GPU-{gpu_id}/{mig_id}/0"

    try:
        # Start orchestrator
        print(f"    Starting orchestrator for {benchmark}...")
        orchestrator_process = start_orchestrator()

        if config["has_sizes"]:
            # Run benchmark for each size
            for size in config["sizes"]:
                print(f"      Running {benchmark} with size {size}...")

                # Run invoker
                result = run_invoker(f"{benchmark}.json", "result.csv")
                if result.returncode != 0:
                    print(
                        f"        Error running invoker for size {size}: {result.stderr}"
                    )
                    print(result.stdout)
                    print(result.stderr)

                result_filename = f"result_{size}.csv"
                result_path = os.path.join(dir_path, result_filename)
                shutil.move("result.csv", result_path)
                print(f"        Results saved to {result_path}")
        else:
            # Run benchmark without sizes
            print(f"      Running {benchmark}...")

            result = run_invoker(f"{benchmark}.json", "result.csv")
            if result.returncode != 0:
                print(f"        Error running invoker: {result.stderr}")
                print(result.stdout)
                print(result.stderr)

            print(f"      Done {benchmark}...")

            result_filename = f"result.csv"
            result_path = os.path.join(dir_path, result_filename)
            shutil.move("result.csv", result_path)

            print(f"        Results saved to {result_path}")

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
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_devices
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # Clean up temporary files
        for temp_file in ["benchmark.json", "result.csv", "orchestrator_output.log"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def main():
    setup_directories()

    gpu_uuid = get_gpuid(device)
    print(f"GPU {device}: {gpu_uuid}")

    for mig_config, profile_id in mig_configs.items():
        print(f"\nRunning benchmarks for configuration: {mig_config}")

        # Run each benchmark
        for benchmark in benchmark_configs.keys():
            print(f"  Running {benchmark}...")

            run_benchmark(gpu_uuid, None, mig_config, benchmark)

        # Cleanup MIG configuration
        if mig_config != "nomig":
            cmd_prt(f"sudo nvidia-smi mig -i {device} -dci")
            cmd_prt(f"sudo nvidia-smi mig -i {device} -dgi")

    # Disable MIG at the end
    cmd_prt(f"sudo nvidia-smi -i {device} -mig 0")


if __name__ == "__main__":
    main()
