import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Function to calculate FWHM
def calculate_fwhm(x, y):
    half_max = max(y) / 2
    # Find indices where y crosses half_max
    greater_than_half = (y > half_max).nonzero()[0]
    left_idx = greater_than_half[0]
    right_idx = greater_than_half[-1]
    # Calculate FWHM
    fwhm = x[right_idx] - x[left_idx]
    return fwhm


def mean_baselane(filedir,params):

    input_dir=filedir

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    # Initialize an empty list to hold the dataframes
    dfs = []
    
    # Loop over the list of CSV files and read each one into a dataframe
    for files in csv_files:
        df = pd.read_csv(files,skiprows=15)
        dfs.append(df)
        
    # Concatenate all dataframes into a single dataframe
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Define the TIME range for filtering
    time_range_start = df['TIME'].iloc[0]
    time_range_end = df['TIME'].iloc[0+params["n_points_pre_wf"]]
    
    # Filter the dataframe for the given TIME range
    filtered_df = final_df[(final_df['TIME'] > time_range_start) & (final_df['TIME'] < time_range_end)]
    
    # Get the list of channels dynamically
    channels = final_df.columns[1:].tolist()  # Exclude the first column which is 'TIME'
    
    # Create subplots
    num_channels = len(channels)
    
    mode_baseline = []
    hwhm_baseline = []
    
    # Loop through each channel and create a histogram
    for i, channel in enumerate(channels):
        # Setup histos
        hist, bins, _ = plt.hist(filtered_df[channel], bins=200, range=(filtered_df[channel].min(), 0.4), alpha=0.7, color='blue', edgecolor='black', density=True)
        
        # Calculate FWHM and mode
        fwhm = calculate_fwhm(bins, hist)
        mean_value = np.average(bins[:-1], weights=hist)
        mode_value = (bins[hist.argmax()] + bins[hist.argmax()+1])/2
        
        mode_baseline.append(mode_value)
        hwhm_baseline.append(fwhm/2)
        

    return mode_baseline,hwhm_baseline
