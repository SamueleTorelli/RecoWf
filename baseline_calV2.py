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


def mean_baselane(final_df,params):

        
    # Define the TIME range for filtering
    time_range_start = final_df['TIME'].iloc[0]
    time_range_end = final_df['TIME'].iloc[0+params["n_points_pre_wf"]]
    
    # Filter the dataframe for the given TIME range
    filtered_df = final_df[(final_df['TIME'] > time_range_start) & (final_df['TIME'] < time_range_end)]
    
    # Get the list of channels dynamically

    channels = final_df.columns[1:-1].tolist()  # Exclude the first column which is 'TIME'
    
    # Create subplots
    num_channels = len(channels)

    
    mode_baseline = []
    hwhm_baseline = []

    #fig, axs = plt.subplots(len(channels), 1, figsize=(10, 15))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, channel in enumerate(channels):
        ax = axs[i%2][i//2]  # Select the subplot
        print(ax)
        # Setup histogram

        minrangehisto = filtered_df[filtered_df[channel]>-np.inf][channel].min()
        maxrangehisto = filtered_df[filtered_df[channel]<np.inf][channel].max()
        

        hist, bins, _ = ax.hist(filtered_df[channel], bins=350, range=(minrangehisto, maxrangehisto),
                                alpha=0.7, color='blue', edgecolor='black', density=True)

        
        # Calculate FWHM and mode
        fwhm = calculate_fwhm(bins, hist)
        mean_value = np.average(bins[:-1], weights=hist)
        mode_value = (bins[hist.argmax()] + bins[hist.argmax()+1])/2
        
        mode_baseline.append(mode_value)
        hwhm_baseline.append(fwhm/2)


        # Draw vertical lines for mode and FWHM
        ax.axvline(mode_value, color='red', linestyle=':', linewidth=2, label=f'Mode: {mode_value:.2e}')
        ax.axvline(mode_value - fwhm / 2, color='purple', linestyle='-.', linewidth=2)
        ax.axvline(mode_value + fwhm / 2, color='purple', linestyle='-.', linewidth=2, label=f'FWHM: {fwhm:.2e}')

        # Add text annotations for the mode and FWHM
        ax.text(mode_value, max(hist) * 0.9, f'Mode: {mode_value:.2e}', color='red', ha='left', fontsize=9)
        ax.text(mode_value + fwhm / 2, max(hist) * 0.8, f'FWHM: {fwhm:.2e}', color='purple', ha='left', fontsize=9)

        
        # Customize subplot
        ax.set_title(f'Channel: {channel}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True)
        
        # Show the plots
    plt.show()
    

    return mode_baseline,hwhm_baseline


