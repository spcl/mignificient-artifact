#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"

    stats = defaultdict(lambda: {"time_us": [], "memory_bytes": []})

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row["op_type"]
            stats[op]["time_us"].append(float(row["time_us"]))
            stats[op]["memory_bytes"].append(float(row["memory_bytes"]))

    for op, data in stats.items():
        n = len(data["time_us"])
        avg_time = sum(data["time_us"]) / n / 1000.0
        avg_bytes = sum(data["memory_bytes"]) / n / 1024.0 / 1024.0
        min_time = min(data["time_us"])
        max_time = max(data["time_us"])
        print(f"--- {op} ({n} samples) ---")
        print(
            f"  time_us:      avg={avg_time:.1f}  min={min_time:.1f}  max={max_time:.1f}"
        )
        print(f"  memory_bytes: avg={avg_bytes:.0f}")
        print()


if __name__ == "__main__":
    main()
