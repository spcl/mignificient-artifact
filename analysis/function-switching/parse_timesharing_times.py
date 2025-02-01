import os
import re
import glob
import pandas as pd

def extract_time_from_file(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'Time:' in line:
                time_str = line.split('Time:')[-1].strip()
                if '.' in time_str:
                    # Assume it's in seconds (float)
                    time_value = int(float(time_str) * 1e6)
                else:
                    # Assume it's in microseconds (integer)
                    time_value = int(time_str)
                times.append(time_value)

    return times[1:] if times else []

def process_files(base_path):
    
    pattern = os.path.join(base_path, "*", "timesharing", "run_worker*.out")
    all_times = []
   
    print(pattern)
    for file_path in glob.glob(pattern):
        print(file_path)
        parts = file_path.split(os.sep)
        app_name = parts[-3]
        worker_number = re.search(r'run_worker(\d+)\.out', file_path)
        if worker_number:
            worker_id = int(worker_number.group(1))
            times = extract_time_from_file(file_path)
            for iteration, time in enumerate(times):
                all_times.append({
                    'APP_NAME': app_name,
                    'worker': worker_id,
                    'iteration': iteration,
                    'time': time
                })
    
    return pd.DataFrame(all_times)

def compute_statistics(df):
    # Convert time to milliseconds and round to one decimal place
    df['time_ms'] = (df['time'] / 1000) #.round(1)
    
    # Compute mean and std dev, excluding iteration 0
    stats = df[df['iteration'] != 0].groupby('APP_NAME')['time_ms'].agg(['mean', 'std']).round(1)
    stats.columns = ['Mean Time (ms)', 'Std Dev Time (ms)']
    return stats.reset_index()

# Usage
base_path = "../../data/function-switching-rtx-4070/function_switching/"  # Replace with the actual base path
result_df = process_files(base_path)

if not result_df.empty:
    print("Sample of extracted data:")
    print(result_df.head())
    print(f"Total rows: {len(result_df)}")
    
    # Compute statistics
    stats_df = compute_statistics(result_df)
    
    print("\nStatistics for each worker:")
    print(stats_df)
    
    # Save the raw data and statistics to CSV files
    result_df.to_csv("extracted_times.csv", index=False)
    stats_df.to_csv("worker_statistics.csv", index=False)
    print("\nRaw data saved to 'extracted_times.csv'")
    print("Statistics saved to 'worker_statistics.csv'")
else:
    print("No data found in the specified files.")
