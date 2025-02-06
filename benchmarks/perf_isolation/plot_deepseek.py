
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read and process the data to DataFrame
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results_deepseek.csv'))

# Create a simplified model name
df['model_short'] = df['model_name'].apply(lambda x: '1.5B' if '1.5B' in x else '7B')

# Calculate average metrics for each model and mode combination
grouped_data = df.groupby(['model_short', 'mode']).agg({
    'time_mean': ['mean', 'std'],
    'time_std': 'mean',
    'time_p99': 'mean'
}).reset_index()

# Flatten column names
grouped_data.columns = ['model_short', 'mode', 'time_mean', 'time_mean_std', 'time_std_mean', 'time_p99_mean']

# Set up the plot style
# plt.style.use('seaborn')
plt.figure(figsize=(12, 6))

# Create bar positions
models = grouped_data['model_short'].unique()
x = np.arange(len(models))
width = 0.35

# Create bars
plt.bar(x - width/2, grouped_data[grouped_data['mode'] == 'mig']['time_mean'], 
        width, label='MIG', color='skyblue', yerr=grouped_data[grouped_data['mode'] == 'mig']['time_std_mean'],
        capsize=5)
plt.bar(x + width/2, grouped_data[grouped_data['mode'] == 'mps']['time_mean'],
        width, label='MPS', color='lightcoral', yerr=grouped_data[grouped_data['mode'] == 'mps']['time_std_mean'],
        capsize=5)

# Customize the plot
plt.xlabel('Model Size')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: MIG vs MPS Modes')
plt.xticks(x, models)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, mode in enumerate(['mig', 'mps']):
    mode_data = grouped_data[grouped_data['mode'] == mode]
    for j, value in enumerate(mode_data['time_mean']):
        plt.text(x[j] + (width/2 if mode == 'mps' else -width/2), value + mode_data['time_std_mean'].iloc[j],
                f'{value:.2f}s', ha='center', va='bottom')

# Adjust layout and display
plt.tight_layout()
plt.savefig("time_deepseek.png")

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for _, row in grouped_data.iterrows():
    print(f"\nModel: {row['model_short']}, Mode: {row['mode'].upper()}")
    print(f"Mean Time: {row['time_mean']:.3f}s ± {row['time_std_mean']:.3f}s")
    print(f"P99 Latency: {row['time_p99_mean']:.3f}s")
