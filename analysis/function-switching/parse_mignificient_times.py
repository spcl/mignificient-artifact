import os
import pandas as pd
import glob

def read_csv_files(base_path):
    # Pattern to match the CSV files
    pattern = os.path.join(base_path, "*/full/*/result.csv")
    
    # List to store individual dataframes
    dfs = []
    
    # Iterate through all matching files
    for file_path in glob.glob(pattern):
        # Extract APP_NAME and MODE from the file path
        parts = file_path.split(os.sep)
        app_name = parts[-4]  # Assuming APP_NAME is 4 levels up from the file
        mode = parts[-2]      # Assuming MODE is 2 levels up from the file
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add APP_NAME and MODE columns
        df['APP_NAME'] = app_name
        df['MODE'] = mode
        
        # Append to the list of dataframes
        dfs.append(df)
    
    # Combine all dataframes
    if dfs:
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Drop rows where iteration is 0
        combined_df = combined_df[combined_df['iteration'] != 0]
        combined_df['time'] = (combined_df['time'] / 1000)
        
        # Reset the index after dropping rows
        combined_df = combined_df.reset_index(drop=True)
        return combined_df
    else:
        print("No CSV files found matching the pattern.")
        return None


def compute_statistics(df):
    # Group by APP_NAME and MODE, then compute mean and std dev for time
    stats = df.groupby(['APP_NAME', 'MODE'])['time'].agg(['mean', 'std']).reset_index()
    
    # Rename columns for clarity
    stats.columns = ['APP_NAME', 'MODE', 'Mean Time (ms)', 'Std Dev Time (ms)']
    stats['Mean Time (ms)'] = stats['Mean Time (ms)'].round(1)
    stats['Std Dev Time (ms)'] = stats['Std Dev Time (ms)'].round(1)
    
    return stats

# Usage
base_path = "../../data/function-switching-rtx-4070/function_switching/"  # Replace with the actual base path
result_df = read_csv_files(base_path)

if result_df is not None:
    #print(result_df.head(100))
    print(f"Total rows: {len(result_df)}")

    stats_df = compute_statistics(result_df)
    
    print("\nStatistics for each APP_NAME and MODE:")
    print(stats_df)
    
    # Optionally, save the statistics to a CSV file
    stats_df.to_csv("statistics_summary.csv", index=False)
    print("\nStatistics have been saved to 'statistics_summary.csv'")
 
    app_order = ['bfs', 'hotspot', 'resnet', 'alexnet', 'vgg19', 'bert']
    mode_order = ['seq_baremetal', 'overlap_baremetal', 'overlap_memcpy_baremetal']
    
    # Prepare the LaTeX table header
    latex_table =  "& ".join(mode_order) + " \\\\\n"
    
    # Generate table rows
    for app in app_order:
        #row = f"{app} "
        row = ""
        for mode in mode_order:
            data = stats_df[(stats_df['APP_NAME'] == app) & (stats_df['MODE'] == mode)]
            if not data.empty:
                mean = data['Mean Time (ms)'].values[0]
                std = data['Std Dev Time (ms)'].values[0]
                row += f"& {mean:.1f} $\\pm$ {std:.1f} "
            else:
                row += "& - "
        row += "\\\\\n"
        latex_table += row

    print(latex_table)
