import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import curve_fit

# Define Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

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

    print(filtered_df)
    
    # Get the list of channels dynamically

    channels = final_df.columns[1:-1].tolist()  # Exclude the first column which is 'TIME'
    print(channels)
    # Create subplots
    num_channels = len(channels)
    
    mode_baseline = []
    hwhm_baseline = []

    cols = 4 if num_channels > 1 else 1
    rows = int(np.ceil(num_channels / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    rangedown = []
    rangeup = []
        
    for i, channel in enumerate(channels):
        minrangehisto = filtered_df[filtered_df[channel]>-np.inf][channel].min()
        maxrangehisto = filtered_df[filtered_df[channel]<np.inf][channel].max()

        rangedown.append(minrangehisto)
        rangeup.append(maxrangehisto)
        
    for i, channel in enumerate(channels):
        ax = axs[i]
        
        print(channel,minrangehisto,maxrangehisto)

        hist, bins, _ = ax.hist(filtered_df[channel], bins=50, range=(min(rangedown), max(rangeup)), #ORIGINAL 120
                                alpha=0.7, color='blue', edgecolor='black')

        # Compute bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Find the index of the maximum bin
        max_bin_idx = np.argmax(hist)
        
        # Define the range of bins to fit (20 bins around max)
        bins_around = 22 #ORIGINAL 16
        fit_start = max(0, max_bin_idx - int(bins_around/2))
        fit_end = min(len(hist), max_bin_idx + int(bins_around/2))
        
        # Select data for fitting
        x_fit = bin_centers[fit_start:fit_end]
        y_fit = hist[fit_start:fit_end]
        
        # Initial guess for Gaussian parameters [Amplitude, Mean, Standard Deviation]
        p0 = [max(y_fit), bin_centers[max_bin_idx], (bins[1] - bins[0]) * 5]

        try:
            # Fit the Gaussian
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
        except:
            popt = [0,0,0]
            
        # Plot the Gaussian fit
        x_smooth = np.linspace(x_fit[0], x_fit[-1], 300)
        y_smooth = gaussian(x_smooth, *popt)
        ax.plot(x_smooth, y_smooth, 'r-', label='Gaussian Fit')

        # Draw vertical lines at fit range limits
        ax.axvline(x_fit[0], color='green', linestyle='--', label='Fit Range')
        ax.axvline(x_fit[-1], color='green', linestyle='--')
        
        """
        # Calculate FWHM and mode
        fwhm = calculate_fwhm(bins, hist)
        mean_value = np.average(bins[:-1], weights=hist)
        mode_value = (bins[hist.argmax()] + bins[hist.argmax()+1])/2
        
        mode_baseline.append(mode_value)
        hwhm_baseline.append(fwhm/2)
        """
        
        mode_baseline.append(popt[1])
        hwhm_baseline.append(popt[2])

        """
        # Draw vertical lines for mode and FWHM
        ax.axvline(mode_value, color='red', linestyle=':', linewidth=2, label=f'Mode: {mode_value:.2e}')
        ax.axvline(mode_value - fwhm / 2, color='purple', linestyle='-.', linewidth=2)
        ax.axvline(mode_value + fwhm / 2, color='purple', linestyle='-.', linewidth=2, label=f'FWHM: {fwhm:.2e}')

        # Add text annotations for the mode and FWHM
        ax.text(mode_value, max(hist) * 0.9, f'Mode: {mode_value:.2e}', color='red', ha='left', fontsize=9)
        ax.text(mode_value + fwhm / 2, max(hist) * 0.8, f'FWHM: {fwhm:.2e}', color='purple', ha='left', fontsize=9)
        """

        #print(f"Fit parameters: Amplitude={popt[0]:.2f}, Mean={popt[1]:.2f}, Sigma={popt[2]:.2f}")

        
        # Customize subplot
        ax.set_title(f'Channel: {channel}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True)
        
        # Show the plots
    plt.show()
    

    return mode_baseline,hwhm_baseline


