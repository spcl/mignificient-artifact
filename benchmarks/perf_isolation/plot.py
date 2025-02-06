import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), 
                 names=['model', 'gen_length', 'mode', 'num_instances', 
                        'instance_id', 'latency'])

# Calculate statistics for each group
grouped_df = df.groupby(['model', 'gen_length', 'mode', 'num_instances']).agg({
    'latency': ['mean', 'std', lambda x: np.percentile(x, 99)]
}).reset_index()

# Rename columns
grouped_df.columns = ['model', 'gen_length', 'mode', 'num_instances', 'mean', 'std', 'p99']

# Get unique values
models = sorted(grouped_df['model'].unique())
gen_lengths = sorted(grouped_df['gen_length'].unique())
instances = sorted(grouped_df['num_instances'].unique())
modes = sorted(grouped_df['mode'].unique())

# Print data shape information for debugging
print(f"Total data points: {len(grouped_df)}")
print(f"Models: {models}")
print(f"Generation lengths: {gen_lengths}")
print(f"Instances: {instances}")
print(f"Modes: {modes}")

# Calculate subplot grid dimensions
n_rows = len(models)
n_cols = len(gen_lengths)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

# Make axes 2D if it's 1D
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Custom colors for modes
mode_colors = {'mig': 'skyblue', 'mps': 'lightgreen'}
bar_width = 0.35

# Create subplots
for i, model in enumerate(models):
    for j, gen_len in enumerate(gen_lengths):
        ax = axes[i, j]
        
        # Filter data for this subplot
        subplot_data = grouped_df[
            (grouped_df['model'] == model) & 
            (grouped_df['gen_length'] == gen_len)
        ]
        
        if len(subplot_data) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center')
            continue
            
        # Set up x-axis positions
        x = np.arange(len(instances))
        
        # Plot bars for each mode
        for idx, mode in enumerate(modes):
            mode_data = subplot_data[subplot_data['mode'] == mode]
            
            if len(mode_data) == 0:
                continue
                
            # Create arrays aligned with instances
            means = []
            stds = []
            p99s = []
            
            for inst in instances:
                inst_data = mode_data[mode_data['num_instances'] == inst]
                if len(inst_data) > 0:
                    means.append(inst_data['mean'].iloc[0])
                    stds.append(inst_data['std'].iloc[0])
                    p99s.append(inst_data['p99'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
                    p99s.append(0)
            
            means = np.array(means)
            stds = np.array(stds)
            p99s = np.array(p99s)
            
            # Plot non-zero bars
            mask = means > 0
            if np.any(mask):
                # Plot bars
                ax.bar(x[mask] + idx*bar_width, 
                      means[mask],
                      bar_width,
                      label=mode,
                      color=mode_colors[mode],
                      alpha=0.8)
                
                # Add error bars
                ax.errorbar(x[mask] + idx*bar_width,
                          means[mask],
                          yerr=[stds[mask], p99s[mask] - means[mask]],
                          fmt='none',
                          color='black',
                          capsize=5)
        
        # Customize subplot
        ax.set_title(f'Model: {model.split("/")[-1]}\nGen Length: {gen_len}')
        ax.set_xticks(x + bar_width/2)
        ax.set_xticklabels(instances)
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('Time (s)')
        
        # Add legend only to first subplot
        if i == 0 and j == 0:
            ax.legend(title='Mode')

# Adjust layout
plt.tight_layout()

# Save the plot in the same folder as the script
output_path = os.path.join(os.path.dirname(__file__), 'benchmark_comparison.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# Print missing combinations for debugging
print("\nMissing combinations:")
for model in models:
    for gen_len in gen_lengths:
        for mode in modes:
            for inst in instances:
                data = grouped_df[
                    (grouped_df['model'] == model) &
                    (grouped_df['gen_length'] == gen_len) &
                    (grouped_df['mode'] == mode) &
                    (grouped_df['num_instances'] == inst)
                ]
                if len(data) == 0:
                    print(f"Missing: model={model}, gen_length={gen_len}, mode={mode}, instances={inst}")
